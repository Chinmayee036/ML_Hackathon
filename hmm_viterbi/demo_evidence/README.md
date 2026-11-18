# Two-laptop rehearsal

Laptop 2 loaded the ten student-test trips and requested trip `00000052`.
Laptop 1 returned HTTP 200 for the prediction, and Laptop 2 rendered the route.

| Metric | Logistic classifier | Classifier + Viterbi |
| --- | ---: | ---: |
| Aligned reference arcs matched | 248 / 262 | 258 / 262 |
| Per-observation accuracy | 94.7% | 98.5% |
| Pairs accepted by the transition rule | 98.08% | 100.00% |

- [Laptop 1 FastAPI request log](laptop1-api-log.png)
- [Laptop 2 route and metrics](laptop2-route-map.png)
- [Results across all ten student-test trips](../student_test_results.csv)

This rehearsal used the logistic regression baseline. The evaluator trip split
and Person 2's final model require a separate run.
