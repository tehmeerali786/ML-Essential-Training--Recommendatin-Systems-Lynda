import pandas as pd
import numpy as np
import os
import webbrowser

# Read dataset into a data table using Pandas
data_table = pd.read_csv("movies.csv", index_col="movie_id")

# Create a web page view of the data for easy viewing
html = data_table.to_html()

# Save the html to a temporary file
with open("movie_list.html", "w") as f:
    f.write(html)
    
# Open the web page in our web browser
full_filename = os.path.abspath("movie_list.html")
webbrowser.open("fill://{}".format(full_filename))