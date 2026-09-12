from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CommandPolicy:
    allowed_executables: set[str] = field(
        default_factory=lambda: {"pytest", "python", "python3", "bash"}
    )
    allow_shell_metacharacters: bool = False
    allow_redirection: bool = False
    allow_chaining: bool = False
    allow_arbitrary_interpreters: bool = False

    def validate(self, command: str | list[str] | None) -> bool:
        if not command:
            return False

        if isinstance(command, str):
            executable = command.strip().split()[0] if command.strip() else ""
            if executable not in self.allowed_executables:
                return False
            if not self.allow_shell_metacharacters and any(token in command for token in ["&&", "||", ";", "|", "<", ">", "*", "?", "$", "`"]):
                return False
            if not self.allow_redirection and any(token in command for token in ["<", ">"]):
                return False
            if not self.allow_chaining and any(token in command for token in ["&&", "||", ";", "|"]):
                return False
            if self.allow_arbitrary_interpreters:
                return True
            return executable not in {"bash", "sh", "zsh", "python", "python3"}

        if not isinstance(command, list):
            return False

        if not command or not command[0]:
            return False

        executable = command[0]
        if executable not in self.allowed_executables:
            return False

        for token in command[1:]:
            if isinstance(token, str) and any(marker in token for marker in ["&&", "||", ";", "|", "<", ">", "*", "?", "$", "`"]):
                return False

        return True
