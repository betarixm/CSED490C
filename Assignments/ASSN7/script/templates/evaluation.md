# Assignment 7. SpMV with JDS format

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
## Results by Data (Not Using Shared Memory)

|{{ results_by_data_fields | join("|") }}|
|{% for _ in range(results_by_data_fields | length) %}:-:|{% endfor %}
{% for row in results_by_data_values -%}
|{{ row | join("|") }}|
{% endfor %}
## Results by Data (Using Shared Memory)

|{{ results_by_data_fields_using_shared_memory | join("|") }}|
|{% for _ in range(results_by_data_fields_using_shared_memory | length) %}:-:|{% endfor %}
{% for row in results_by_data_values_using_shared_memory -%}
|{{ row | join("|") }}|
{% endfor %}
