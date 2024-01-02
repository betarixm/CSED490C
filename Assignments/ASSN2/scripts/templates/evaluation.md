# Assignment 2. Tiled Matrix Multiplication

> Machine generated report

## Environment

### CPU

{% for key, value in cpu_info -%}
- {{ key }}: {{ value }}
{% endfor %}
### GPU

{% for key, value in gpu_info -%}
- {{ key }}: {{ value }}
{% endfor %}
### Others

{% for key, value in host_info -%}
- {{ key }}: {{ value }}
{% endfor %}
## Results by Data

|{{ results_by_data_fields | join("|") }}|
|{% for _ in range(results_by_data_fields | length) %}:-:|{% endfor %}
{% for row in results_by_data_values -%}
|{{ row | join("|") }}|
{% endfor %}
## Results by Tile Width

|{{ results_by_tile_width_fields | join("|") }}|
|{% for _ in range(results_by_tile_width_fields | length) %}:-:|{% endfor %}
{% for row in results_by_tile_width_values -%}
|{{ row | join("|") }}|
{% endfor %}