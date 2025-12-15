from typing import Any

import jinja2


def render_prompt_template(
    template_name: str, context: dict[str, Any], template_dir: str = "prompts"
) -> str:
    """
    Loads a Jinja2 template and renders it with the provided context.

    Args:
        template_name: Name of the template file
        context: Dictionary of context variables to render
        template_dir: Directory containing templates (default: "prompts")
    """
    template_loader = jinja2.FileSystemLoader(searchpath=template_dir)
    template_env = jinja2.Environment(
        loader=template_loader, undefined=jinja2.StrictUndefined
    )
    template = template_env.get_template(template_name)
    return template.render(**context)
