import os
import pandas as pd
from contextlib import suppress
from src.start import classifier

queries_answers_df = pd.read_csv(os.path.join(os.getcwd(), "data", "queries_with_answers.csv"), sep="\t")
queries_answers_dicts = queries_answers_df.to_dict(orient="records")

results = []
for num, d in enumerate(queries_answers_df.to_dict(orient="records")):
    with suppress(TypeError):
        val_d = classifier.t5_validate(d["Query"], d["FastAnswerText"], 0.0)
        res_dict = {**d, **val_d}
        results.append(res_dict)
    print(num)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(os.getcwd(), "results", "test_result.csv"), sep="\t", index=False)