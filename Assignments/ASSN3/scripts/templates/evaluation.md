# Assignment 3. Vector Add with CUDA Streams and Pinned Memory

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
## Results by the Number of Streams

|{{ results_by_number_of_streams_fields | join("|") }}|
|{% for _ in range(results_by_number_of_streams_fields | length) %}:-:|{% endfor %}
{% for row in results_by_number_of_streams_values -%}
|{{ row | join("|") }}|
{% endfor %}