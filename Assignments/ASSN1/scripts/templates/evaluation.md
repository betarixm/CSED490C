# Assignment 1. Vector Addition

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
## Elapsed Times

|{{ fields | join("|") }}|
|{% for _ in range(fields | length) %}:-:|{% endfor %}
{% for row in results -%}
|{{ row | join("|") }}|
{% endfor %}