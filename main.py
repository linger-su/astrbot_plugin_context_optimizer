"""
上下文优化器插件 - 核心设计：

默认模式：每次只给 LLM 发送最新消息，极致省 token。
压缩模式：用 AstrBot 已配置的 LLM provider 对历史做摘要压缩。
自动展开：LLM 回复表现出困惑时，自动补入压缩后的历史上下文。
Token 统计：每次对话显示实际消耗和节省的 token 数。
"""

import time
import re
from typing import Optional, List, Dict
from collections import deque

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import ProviderRequest
from astrbot.api import logger, AstrBotConfig
from astrbot.core.message.components import Plain
from astrbot.core.provider.entites import LLMResponse

from .compressor import Compressor


@register(
    "astrbot_plugin_context_optimizer",
    "AutoClaw",
    "极致省token上下文优化器：默认只发最新消息，LLM看不懂时自动补上下文",
    "2.2.0",
)
class ContextOptimizerPlugin(Star):
    """
    上下文优化器 v2.2

    核心策略：
    1. 默认只发送最新用户消息 + system prompt → 极致省 token
    2. 内存中维护完整对话历史 + 压缩摘要
    3. 检测到 LLM 回复是困惑/澄清请求时，自动展开上下文
    4. 支持用 AstrBot 已配置的 LLM provider 做智能压缩
    5. 每次对话显示 token 统计：发送/返回/消耗/节省
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        # --- 配置项 ---
        self.mode = self.config.get("context_mode", "minimal")
        self.preserve_turns = int(self.config.get("preserve_recent_turns", 1))
        self.compress_ratio = float(self.config.get("compress_ratio", 0.3))
        self.max_compressed_tokens = int(self.config.get("max_compressed_tokens", 200))
        self.confusion_threshold = float(self.config.get("confusion_threshold", 0.5))
        self.max_expand_retries = int(self.config.get("max_expand_retries", 2))
        self.remove_tool_calls = self.config.get("remove_tool_calls", True)
        self.remove_tool_results = self.config.get("remove_tool_results", True)
        self.log_optimization = self.config.get("log_optimization", False)

        # --- 压缩方式 ---
        self.compress_method = self.config.get("compress_method", "tfidf")
        self.compress_provider_id = self.config.get("compress_provider_id", "")

        # --- Token 统计 ---
        self.show_token_stats = self.config.get("show_token_stats", False)

        # --- 内部状态 ---
        self._compressor = Compressor()
        self._llm_provider = None

        # 对话历史（内存）
        self.max_history = int(self.config.get("max_history_entries", 200))
        self._history: deque = deque(maxlen=self.max_history)

        # 压缩后的历史摘要
        self._compressed_summary: str = ""

        # 多会话状态
        self._session_state: Dict[str, dict] = {}

        # 每个请求的 token 快照（用于 on_decorating_result 读取）
        self._request_tokens: Dict[str, dict] = {}

        # 统计
        self._stats = {
            "total_requests": 0,
            "minimal_sends": 0,
            "expand_triggers": 0,
            "llm_compress_calls": 0,
            "llm_compress_errors": 0,
            "tfidf_compress_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_saved_tokens": 0,
        }

        logger.info(
            f"上下文优化器 v2.2 已初始化 | "
            f"模式={self.mode} 压缩={self.compress_method} "
            f"Token统计={'开' if self.show_token_stats else '关'}"
        )

        self._log_available_providers()

    def _log_available_providers(self):
        """打印所有可用的 provider"""
        try:
            providers = self.context.get_providers()
            if providers:
                names = []
                for p in providers:
                    name = getattr(p, 'name', None) or getattr(p, 'provider_name', None) or str(p)
                    names.append(name)
                logger.info(
                    f"[上下文优化器] 可用 LLM Provider: {', '.join(names)} | "
                    f"将 compress_provider_id 设为上述名称之一，或留空使用默认"
                )
        except Exception as e:
            logger.debug(f"[上下文优化器] 获取 Provider 列表失败: {e}")

    # ========== Provider 获取 ==========

    def _get_compress_provider(self):
        """获取用于压缩的 LLM provider"""
        if self._llm_provider:
            return self._llm_provider

        try:
            if self.compress_provider_id:
                self._llm_provider = self.context.get_provider(self.compress_provider_id)
            else:
                self._llm_provider = self.context.get_provider()

            if self._llm_provider:
                provider_name = getattr(self._llm_provider, 'name', 'unknown')
                logger.info(f"上下文优化器：已获取 LLM provider - {provider_name}")
            else:
                logger.warning("上下文优化器：未找到可用 LLM provider，回退到 TF-IDF")

            return self._llm_provider
        except Exception as e:
            logger.warning(f"上下文优化器：获取 LLM provider 失败: {e}")
            return None

    async def _llm_compress(self, text: str) -> Optional[str]:
        """用 LLM 做摘要压缩"""
        provider = self._get_compress_provider()
        if not provider or not text or not text.strip():
            return None

        # 截断过长输入
        max_input_tokens = self.max_compressed_tokens * 5
        input_tokens = self._compressor.estimate_tokens(text)
        if input_tokens > max_input_tokens:
            ratio = max_input_tokens / input_tokens
            text = text[:int(len(text) * ratio)]

        compress_prompt = f"""请对以下对话历史进行压缩总结，要求：
1. 保留关键信息、重要事实和核心观点
2. 去掉重复、无关、冗余的内容
3. 使用简洁的语言，尽量精炼
4. 目标长度：约{self.max_compressed_tokens}个token以内
5. 保留人名、数字、专有名词等关键实体
6. 用中文总结

对话历史：
{text}

压缩后的摘要："""

        try:
            response = await provider.chat([{
                "role": "user",
                "content": compress_prompt,
            }])

            if response and hasattr(response, 'completion'):
                result = response.completion
            elif response and isinstance(response, str):
                result = response
            elif response and isinstance(response, dict):
                result = response.get("content", "") or response.get("text", "")
            else:
                result = str(response) if response else None

            if result and result.strip():
                self._stats["llm_compress_calls"] += 1
                return result.strip()
            return None

        except Exception as e:
            self._stats["llm_compress_errors"] += 1
            logger.warning(f"上下文优化器：LLM 压缩调用失败: {e}")
            return None

    # ========== 核心 Hook ==========

    @filter.on_llm_request()
    async def optimize_context(self, event: AstrMessageEvent, req: ProviderRequest):
        """核心拦截器：在 LLM 请求前优化上下文"""
        try:
            self._stats["total_requests"] += 1

            if not req.contexts:
                return

            session_key = self._get_session_key(event)
            state = self._session_state.setdefault(
                session_key, {"sent_minimal": False, "expand_count": 0}
            )

            # 记录历史（在修改之前）
            self._record_to_history(req.contexts)

            # 记录优化前的 token 估算
            original_tokens = self._estimate_tokens(req)
            original_count = len(req.contexts)

            # --- full 模式 ---
            if self.mode == "full":
                req.contexts = self._basic_clean(req.contexts)
                self._save_request_snapshot(session_key, original_tokens, req)
                logger.info(f"[上下文优化] full模式: {original_count}条 → {len(req.contexts)}条")
                return

            # --- 检测上一轮困惑 ---
            was_confused = self._detect_confusion(req.contexts)

            if was_confused and state["sent_minimal"] and state["expand_count"] < self.max_expand_retries:
                state["expand_count"] += 1
                req.contexts = await self._build_expanded_context(req)
                state["sent_minimal"] = False
                self._stats["expand_triggers"] += 1
                logger.info(f"[上下文优化] 展开上下文(第{state['expand_count']}次): {original_count}条 → {len(req.contexts)}条")
            elif self.mode == "minimal":
                req.contexts = self._build_minimal_context(req)
                state["sent_minimal"] = True
                state["expand_count"] = 0
                self._stats["minimal_sends"] += 1
                logger.info(f"[上下文优化] 极简模式: {original_count}条 → {len(req.contexts)}条")
            else:
                req.contexts = await self._build_balanced_context(req)
                logger.info(f"[上下文优化] 均衡模式: {original_count}条 → {len(req.contexts)}条")

            # 保存快照
            self._save_request_snapshot(session_key, original_tokens, req)

        except Exception as e:
            logger.warning(f"[上下文优化器] optimize_context 异常: {e}")

    def _save_request_snapshot(self, session_key: str, original_tokens: int, req: ProviderRequest):
        """保存本次请求的 token 快照"""
        optimized_tokens = self._estimate_tokens(req)
        saved = max(0, original_tokens - optimized_tokens)
        self._request_tokens[session_key] = {
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "saved_tokens": saved,
        }

    @filter.on_llm_response()
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse):
        """捕获 LLM 响应，提取实际 token 用量"""
        if not self.show_token_stats:
            return

        try:
            session_key = self._get_session_key(event)
            snap = self._request_tokens.get(session_key, {})

            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            try:
                completion = resp.raw_completion
                if completion and hasattr(completion, 'usage') and completion.usage:
                    usage = completion.usage
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
                    completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
                    total_tokens = getattr(usage, 'total_tokens', 0) or 0
            except Exception:
                pass

            # 更新全局统计
            self._stats["total_prompt_tokens"] += prompt_tokens
            self._stats["total_completion_tokens"] += completion_tokens
            self._stats["total_saved_tokens"] += snap.get("saved_tokens", 0)

            # 构建统计信息
            saved = snap.get("saved_tokens", 0)
            stats_parts = []
            if prompt_tokens > 0:
                stats_parts.append(f"发送:{prompt_tokens}")
            if completion_tokens > 0:
                stats_parts.append(f"返回:{completion_tokens}")
            if total_tokens > 0:
                stats_parts.append(f"消耗:{total_tokens}")
            if saved > 0:
                stats_parts.append(f"节省:~{saved}")

            token_msg = f"\n📊 Token | {' | '.join(stats_parts)}" if stats_parts else ""

            # 存到 event 上供 on_decorating_result 使用
            if token_msg:
                event._ctx_opt_token = token_msg

            # 清理快照
            self._request_tokens.pop(session_key, None)

        except Exception as e:
            logger.debug(f"[上下文优化器] on_llm_resp 异常: {e}")

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        """在回复末尾追加 token 统计"""
        if not self.show_token_stats:
            return

        try:
            token_msg = getattr(event, '_ctx_opt_token', None)
            if not token_msg:
                return

            result = event.get_result()
            if result and hasattr(result, 'chain'):
                result.chain.append(Plain(token_msg))
        except Exception as e:
            logger.debug(f"[上下文优化器] on_decorating_result 异常: {e}")

    # ========== 上下文构建 ==========

    def _build_minimal_context(self, req: ProviderRequest) -> list:
        """极简模式：只保留 system_prompt + 最新用户消息"""
        contexts = req.contexts
        if not contexts:
            return contexts

        cleaned = self._basic_clean(contexts)

        last_user_idx = -1
        for i in range(len(cleaned) - 1, -1, -1):
            if cleaned[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx == -1:
            return cleaned

        result = []
        for i, ctx in enumerate(cleaned):
            if ctx.get("role") == "system":
                result.append(ctx)
            elif i >= last_user_idx:
                result.append(ctx)

        return result if result else cleaned[-1:]

    async def _build_balanced_context(self, req: ProviderRequest) -> list:
        """均衡模式：保留最近N轮 + 压缩更早的历史"""
        contexts = req.contexts
        if not contexts:
            return contexts

        cleaned = self._basic_clean(contexts)
        system_msgs = [ctx for ctx in cleaned if ctx.get("role") == "system"]
        chat_msgs = [ctx for ctx in cleaned if ctx.get("role") != "system"]

        if not chat_msgs:
            return cleaned

        user_count = 0
        protected_start = len(chat_msgs)
        for i in range(len(chat_msgs) - 1, -1, -1):
            if chat_msgs[i].get("role") == "user":
                user_count += 1
                if user_count >= self.preserve_turns:
                    protected_start = i
                    break

        older_msgs = chat_msgs[:protected_start]
        newer_msgs = chat_msgs[protected_start:]
        result = list(system_msgs)

        if older_msgs:
            older_text = self._messages_to_text(older_msgs)
            if older_text and self._compressor.estimate_tokens(older_text) > 50:
                compressed = await self._compress_text(older_text)
                if compressed and compressed.strip():
                    result.append({"role": "system", "content": f"[对话历史摘要] {compressed}"})

        result.extend(newer_msgs)
        return result

    async def _build_expanded_context(self, req: ProviderRequest) -> list:
        """展开模式：当 LLM 困惑时，补入更完整的历史"""
        contexts = req.contexts
        cleaned = self._basic_clean(contexts)
        system_msgs = [ctx for ctx in cleaned if ctx.get("role") == "system"]
        chat_msgs = [ctx for ctx in cleaned if ctx.get("role") != "system"]
        result = list(system_msgs)

        if self._history:
            all_text = "\n".join(f"[{e['role']}] {e['text']}" for e in self._history)
            if all_text:
                compressed = await self._compress_text(all_text, ratio_boost=0.2)
                if compressed and compressed.strip():
                    result.append({"role": "system", "content": f"[完整对话上下文] {compressed}"})

        result.extend(chat_msgs)
        return result

    # ========== 压缩调度 ==========

    async def _compress_text(self, text: str, ratio_boost: float = 0) -> str:
        """统一压缩入口"""
        if self.compress_method in ("llm", "llm+tfidf"):
            result = await self._llm_compress(text)
            if result:
                return result
            if self.compress_method == "llm":
                logger.debug("上下文优化器：LLM 压缩失败，回退到 TF-IDF")

        ratio = min(0.9, self.compress_ratio + ratio_boost)
        self._stats["tfidf_compress_calls"] += 1
        return self._compressor.compress(text, ratio=ratio, max_tokens=self.max_compressed_tokens)

    # ========== 工具方法 ==========

    def _basic_clean(self, contexts: list) -> list:
        """基础清洗：过滤工具调用/结果"""
        cleaned = []
        for ctx in contexts:
            role = ctx.get("role", "")
            if self.remove_tool_results and role == "tool":
                continue
            if self.remove_tool_calls and role == "assistant":
                if ctx.get("tool_calls"):
                    new_ctx = {k: v for k, v in ctx.items() if k != "tool_calls"}
                    if new_ctx.get("content"):
                        cleaned.append(new_ctx)
                    continue
            cleaned.append(ctx)
        return cleaned

    def _record_to_history(self, contexts: list):
        """将对话内容记录到内存历史"""
        for ctx in contexts:
            role = ctx.get("role", "")
            content = ctx.get("content", "")
            if content and isinstance(content, str) and role in ("user", "assistant"):
                text = content.strip()
                if len(text) > 2:
                    self._history.append({"role": role, "text": text, "ts": time.time()})

        if len(self._history) % 10 == 0 and self._history:
            self._update_compressed_summary()

    def _update_compressed_summary(self):
        """更新压缩后的历史摘要"""
        all_text = "\n".join(e["text"] for e in self._history)
        if all_text:
            self._compressed_summary = self._compressor.compress(
                all_text, ratio=self.compress_ratio, max_tokens=self.max_compressed_tokens
            )

    def _detect_confusion(self, contexts: list) -> bool:
        """检测上一轮 assistant 回复是否是困惑/澄清请求"""
        if not contexts:
            return False
        for ctx in reversed(contexts):
            if ctx.get("role") == "assistant":
                content = ctx.get("content", "")
                if content and isinstance(content, str):
                    score = self._compressor.confusion_score(content)
                    if score >= self.confusion_threshold:
                        return True
                break
        return False

    def _messages_to_text(self, messages: list) -> str:
        parts = []
        for ctx in messages:
            role = ctx.get("role", "unknown")
            content = ctx.get("content", "")
            if content and isinstance(content, str):
                parts.append(f"[{role}] {content}")
        return "\n".join(parts)

    def _get_session_key(self, event: AstrMessageEvent) -> str:
        try:
            if hasattr(event, 'session_id') and event.session_id:
                return str(event.session_id)
            if hasattr(event, 'get_session_id'):
                return str(event.get_session_id())
        except:
            pass
        return "default"

    def _estimate_tokens(self, req: ProviderRequest) -> int:
        total = 0
        if req.system_prompt:
            total += self._compressor.estimate_tokens(req.system_prompt)
        if req.contexts:
            for ctx in req.contexts:
                content = ctx.get("content", "")
                if isinstance(content, str):
                    total += self._compressor.estimate_tokens(content)
        if hasattr(req, "prompt") and req.prompt:
            total += self._compressor.estimate_tokens(req.prompt)
        return total

    # ========== 生命周期 ==========

    async def terminate(self):
        logger.info(
            f"上下文优化器 v2.2 已卸载 | "
            f"请求{self._stats['total_requests']}次, "
            f"极简{self._stats['minimal_sends']}次, "
            f"展开{self._stats['expand_triggers']}次 | "
            f"累计发送{self._stats['total_prompt_tokens']}tokens, "
            f"返回{self._stats['total_completion_tokens']}tokens, "
            f"节省约{self._stats['total_saved_tokens']}tokens"
        )
