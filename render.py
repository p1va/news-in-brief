from typing import Any

import jinja2


def render_prompt_template(template_name: str, context: dict[str, Any]) -> str:
    """
    Loads a Jinja2 template and renders it with the provided context.
    """
    template_loader = jinja2.FileSystemLoader(searchpath="prompts")
    template_env = jinja2.Environment(
        loader=template_loader, undefined=jinja2.StrictUndefined
    )
    template = template_env.get_template(template_name)
    return template.render(**context)
