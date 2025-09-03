"""
UI компоненты для Справочника соединений
"""

from .styles import inject_styles
from .utils import (
    truncate_description, format_chemical_formula, safe_get_value,
    format_mass, create_external_links, create_pills_list,
    calculate_pagination_info, format_search_query, validate_search_params,
    get_display_name, create_metric_html, create_stats_html
)
from .cards import (
    render_metabolite_card, render_enzyme_card, render_protein_card,
    render_carbohydrate_card, render_lipid_card
)
from .details import (
    show_metabolite_details, show_enzyme_details, show_protein_details,
    show_carbohydrate_details, show_lipid_details
)
from .search_form import (
    render_search_form, handle_search_form, render_pagination,
    render_view_toggle, render_results_header, render_close_details_buttons
)
from .visualization_3d import (
    render_3d_structure, display_molecule_properties, render_2d_structure,
    check_dependencies, install_instructions
)

__all__ = [
    'inject_styles',
    'truncate_description', 'format_chemical_formula', 'safe_get_value',
    'format_mass', 'create_external_links', 'create_pills_list',
    'calculate_pagination_info', 'format_search_query', 'validate_search_params',
    'get_display_name', 'create_metric_html', 'create_stats_html',
    'render_metabolite_card', 'render_enzyme_card', 'render_protein_card',
    'render_carbohydrate_card', 'render_lipid_card',
    'show_metabolite_details', 'show_enzyme_details', 'show_protein_details',
    'show_carbohydrate_details', 'show_lipid_details',
    'render_search_form', 'handle_search_form', 'render_pagination',
    'render_view_toggle', 'render_results_header', 'render_close_details_buttons',
    'render_3d_structure', 'display_molecule_properties', 'render_2d_structure',
    'check_dependencies', 'install_instructions'
]
