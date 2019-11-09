### ML Tools and Software:

- **Anaconda** - super platform for ML that comes with an jupyter ide and all the libs out of the box, and provides all kinds of tools for ML

 *installation:*
 
 go to [anaconda.com](http://www.anaconda.com/download), download and install
 (try selecting the "add to path" option, despite the warning)
 
 - **jupyter** - a ML IDE

after installing anaconda try typing in cmd `jupyter notebook`, if not recognized look for this option in the start menu,and open from there. a new browser window will automatically open.
the page shows computer files explorer, go to some dir like desktop and select "New notebook"
give it a name at the top. this will create a name.ipynb file in the selected location.

go to [kaggle.com](https://www.kaggle.com/) to get some data set to work with.
search for some data, i.e. "video game sales", select a set and download it

put the downloaded csv file in same dir as the .ipynb file is. 
return to jupyter and input this code and click run.
```
import pandas as pd
df= pd.read_csv('vgsales.csv')
df
```
this will output the dataframe (content of the file)
when clicking on run btn, the In[] section that is hightlighted in blue is that one that will execute. the others wont
To run all cells, go to menu `Cell`->`Run All`

**shortcuts:**

press on `H` key, a popup will open with shortcuts list

press `D` twice to delete a cell (the blue hightlighted one)

to show possible functions on object, write objectname. and hit `TAB`

to read info about a function hover over the func name in the input cell and hit `shift`+ tab 
