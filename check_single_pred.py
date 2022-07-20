
import pandas as pd
from model import model

file_path = r'/home/victore/Downloads/one_window_for_demo.csv'
dir_path = '/home/victore/Downloads'

pipe = model()
pipe.load(dir_path)

data = pd.read_csv(file_path)

# pipe.model.fit([data, data], [1.0, 0.0])

pipe.predict(data)

