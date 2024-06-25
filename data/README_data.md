# Stochastic Games for User Journeys - Data Description

The contained `data_io.list` file is a processed version of [GrepS](https://zenodo.org/records/6962413/files/data.csv?download=1), containing a list of IO-traces as described in the paper.
It was produced in `io_alergia_greps.py` and stored using Python's `pickle`:
```python
import pickle
from journepy.src.preprocessing.greps import preprocessed_log
# Load files and filter
filtered_log = preprocessed_log(DATA_PATH+'data.csv', include_loggin=False) # also discards task-event log-in
# Load actor information
with open(DATA_PATH+'activities_greps.xml') as f:
    data = f.read()
actors = json.loads(data)
# Transform into IO format
data_environment = get_data(actors, filtered_log)
# Save to disk
with open(DATA_PATH+'data_io.list', 'wb') as f:
    pickle.dump(data_environment, f)
```