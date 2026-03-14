from __future__ import annotations

import re
from typing import Iterator, Optional

from smolagents.agent_types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from smolagents.agents import ActionStep, MultiStepAgent
from smolagents.memory import MemoryStep

from .types import MessageContent, StreamlitMessage


def pull_messages_from_step_streamlit(step_log: MemoryStep) -> Iterator[StreamlitMessage]:
    """
    Convert one agent step into a sequence of `StreamlitMessage` objects.

    The structure mirrors the Gradio implementation but is tailored for
    Streamlit's chat API and our custom `StreamlitMessage` type.
    """
    if isinstance(step_log, ActionStep):
        # High‑level step header (e.g. "Step 1")
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else ""
        yield StreamlitMessage(role="assistant", content=f"**{step_number}**")

        # Model "thoughts" / reasoning text from the LLM, cleaned from stray markers.
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            model_output = step_log.model_output.strip()
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)
            model_output = model_output.strip()
            if model_output:
                yield StreamlitMessage(role="assistant", content=model_output)

        # Tool calls and their associated logs / errors.
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            parent_id = f"call_{len(step_log.tool_calls)}"

            args = first_tool_call.arguments
            if isinstance(args, dict):
                content = str(args.get("answer", str(args)))
            else:
                content = str(args).strip()

            if used_code:
                # Normalize any stray code markers so the content is a clean python block.
                content = re.sub(r"```.*?\n", "", content)
                content = re.sub(r"\s*<end_code>\s*", "", content).strip()
                if not content.startswith("```python"):
                    content = f"```python\n{content}\n```"

            yield StreamlitMessage(
                role="assistant",
                content=content,
                metadata={"title": f"🛠️ Used tool {first_tool_call.name}", "id": parent_id, "status": "done"},
            )

            if hasattr(step_log, "observations") and step_log.observations is not None and step_log.observations.strip():
                log_content = re.sub(r"^Execution logs:\s*", "", step_log.observations.strip()).strip()
                if log_content:
                    yield StreamlitMessage(
                        role="assistant",
                        content=log_content,
                        metadata={"title": "📝 Execution Logs", "parent_id": parent_id, "status": "done"},
                    )

            if hasattr(step_log, "error") and step_log.error is not None:
                yield StreamlitMessage(
                    role="assistant",
                    content=str(step_log.error),
                    metadata={"title": "💥 Error", "parent_id": parent_id, "status": "done"},
                )
        elif hasattr(step_log, "error") and step_log.error is not None:
            # Error that is not attached to a specific tool call.
            yield StreamlitMessage(role="assistant", content=str(step_log.error), metadata={"title": "💥 Error"})

        # Per‑step footnote with token and timing information.
        step_footnote = f"{step_number}"
        if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
            step_footnote += (
                f" | Input-tokens:{step_log.input_token_count:,}"
                f" | Output-tokens:{step_log.output_token_count:,}"
            )
        if hasattr(step_log, "duration") and step_log.duration:
            step_footnote += f" | Duration: {round(float(step_log.duration), 2)}"

        yield StreamlitMessage(
            role="assistant",
            content=f'<span style="color: #9aa0a6; font-size: 12px;">{step_footnote}</span>',
        )


def stream_to_streamlit_messages(
    agent: MultiStepAgent,
    task: str,
    *,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
) -> Iterator[StreamlitMessage]:
    """
    Run the agent with streaming enabled and yield `StreamlitMessage` chunks.

    This function is UI‑agnostic: it only turns agent steps into friendly
    message objects, leaving actual rendering to the caller.
    """
    last_log: Optional[MemoryStep] = None

    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
        last_log = step_log

        # If the model exposes token counts, attach them to ActionStep instances.
        if hasattr(agent, "model") and hasattr(agent.model, "last_input_token_count"):
            if isinstance(step_log, ActionStep):
                step_log.input_token_count = agent.model.last_input_token_count
                step_log.output_token_count = agent.model.last_output_token_count

        for message in pull_messages_from_step_streamlit(step_log):
            yield message

    if last_log is None:
        return

    # Turn the final agent output into a last message for the chat.
    final_answer = handle_agent_output_types(last_log)
    if isinstance(final_answer, AgentText):
        yield StreamlitMessage(role="assistant", content=f"**Final answer:**\n\n{final_answer.to_string()}\n")
    elif isinstance(final_answer, AgentImage):
        yield StreamlitMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
            metadata={"title": "🖼️ Final answer"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield StreamlitMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
            metadata={"title": "🔊 Final answer"},
        )
    else:
        yield StreamlitMessage(role="assistant", content=f"**Final answer:** {str(final_answer)}")


__all__ = ["MessageContent", "StreamlitMessage", "pull_messages_from_step_streamlit", "stream_to_streamlit_messages"]

