"""语音命令匹配的单元测试"""

from voxy.commands import match_command


class TestExactMatch:
    def test_exact_match(self):
        cmd_map = {"发送": "keys:Return", "撤销": "keys:ctrl+z"}
        assert match_command("发送", cmd_map) == ("发送", "keys:Return")
        assert match_command("撤销", cmd_map) == ("撤销", "keys:ctrl+z")

    def test_no_match(self):
        cmd_map = {"发送": "keys:Return"}
        assert match_command("今天天气不错", cmd_map) is None

    def test_strip_whitespace(self):
        cmd_map = {"发送": "keys:Return"}
        assert match_command("  发送  ", cmd_map) == ("发送", "keys:Return")

    def test_strip_trailing_punctuation(self):
        cmd_map = {"发送": "keys:Return"}
        assert match_command("发送。", cmd_map) == ("发送", "keys:Return")
        assert match_command("发送！", cmd_map) == ("发送", "keys:Return")
        assert match_command("发送.", cmd_map) == ("发送", "keys:Return")
        assert match_command("发送，", cmd_map) == ("发送", "keys:Return")
        assert match_command("发送？", cmd_map) == ("发送", "keys:Return")

    def test_empty_text(self):
        cmd_map = {"发送": "keys:Return"}
        assert match_command("", cmd_map) is None
        assert match_command("   ", cmd_map) is None


class TestEmptyMap:
    def test_empty_map_skips(self):
        assert match_command("发送", {}) is None

    def test_none_like_empty(self):
        assert match_command("发送", {}) is None


class TestFuzzyMatch:
    def test_fuzzy_match_similar(self):
        cmd_map = {"撤销": "keys:ctrl+z"}
        # "撤消" vs "撤销" — 一字之差，ratio 应该很高
        result = match_command("撤消", cmd_map, fuzzy_threshold=0.5)
        assert result == ("撤销", "keys:ctrl+z")

    def test_fuzzy_below_threshold(self):
        cmd_map = {"撤销": "keys:ctrl+z"}
        # 完全不相关的文本
        assert match_command("今天天气真好", cmd_map, fuzzy_threshold=0.8) is None

    def test_fuzzy_disabled_by_default(self):
        cmd_map = {"撤销": "keys:ctrl+z"}
        # 默认 threshold=0，不走模糊匹配
        assert match_command("撤消", cmd_map) is None

    def test_exact_match_takes_priority_over_fuzzy(self):
        cmd_map = {"发送": "keys:Return", "发送消息": "keys:ctrl+Return"}
        # 精确匹配应该优先
        assert match_command("发送", cmd_map, fuzzy_threshold=0.5) == ("发送", "keys:Return")


class TestComboActions:
    def test_combo_action_string(self):
        cmd_map = {"删除": "keys:ctrl+a|keys:BackSpace"}
        result = match_command("删除", cmd_map)
        assert result == ("删除", "keys:ctrl+a|keys:BackSpace")

    def test_shell_action(self):
        cmd_map = {"提交代码": "shell:git add -A && git commit"}
        result = match_command("提交代码", cmd_map)
        assert result == ("提交代码", "shell:git add -A && git commit")

    def test_text_action(self):
        cmd_map = {"我的邮箱": "text:user@example.com"}
        result = match_command("我的邮箱", cmd_map)
        assert result == ("我的邮箱", "text:user@example.com")


class TestContextOverride:
    """窗口上下文覆盖测试"""

    def setup_method(self):
        self.cmd_map = {"发送": "keys:Return", "换行": "keys:Return"}
        self.context = {
            "foot": {"发送": "keys:Return", "换行": "text:\\|keys:Return"},
            "firefox": {"发送": "keys:ctrl+Return"},
            "vivaldi": {"发送": "keys:ctrl+Return"},
        }

    def test_context_overrides_default(self):
        result = match_command(
            "发送", self.cmd_map,
            window_class="firefox", context=self.context,
        )
        assert result == ("发送", "keys:ctrl+Return")

    def test_context_preserves_unoverridden(self):
        # firefox context 只覆盖"发送"，"换行"应保持默认
        result = match_command(
            "换行", self.cmd_map,
            window_class="firefox", context=self.context,
        )
        assert result == ("换行", "keys:Return")

    def test_case_insensitive_contains_match(self):
        # "vivaldi" 应匹配 "vivaldi-stable"
        result = match_command(
            "发送", self.cmd_map,
            window_class="vivaldi-stable", context=self.context,
        )
        assert result == ("发送", "keys:ctrl+Return")

    def test_case_insensitive_pattern(self):
        # context key "Firefox" (大写) 也应匹配 window_class "firefox"
        context = {"Firefox": {"发送": "keys:ctrl+Return"}}
        result = match_command(
            "发送", self.cmd_map,
            window_class="firefox", context=context,
        )
        assert result == ("发送", "keys:ctrl+Return")

    def test_no_matching_context_uses_default(self):
        result = match_command(
            "发送", self.cmd_map,
            window_class="kitty", context=self.context,
        )
        assert result == ("发送", "keys:Return")

    def test_no_window_class_uses_default(self):
        result = match_command(
            "发送", self.cmd_map,
            window_class=None, context=self.context,
        )
        assert result == ("发送", "keys:Return")

    def test_empty_context_uses_default(self):
        result = match_command(
            "发送", self.cmd_map,
            window_class="firefox", context={},
        )
        assert result == ("发送", "keys:Return")

    def test_foot_context_override(self):
        result = match_command(
            "换行", self.cmd_map,
            window_class="foot", context=self.context,
        )
        assert result == ("换行", "text:\\|keys:Return")
