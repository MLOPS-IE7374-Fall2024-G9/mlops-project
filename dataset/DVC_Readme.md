# Data Version Control (DVC)

This is used to track our changes made in the data files for this project. The actual data itself sits in a Google Cloud bucket. This tool helps us track our data alongside our code. 

## Getting started
If developing on this project, just install the `requirements_dev.txt` by running `pip install -r requirements_dev.txt` AFTER activating your development environment. It should install the DVC package (along with all other dependencies).

## Configure Google Cloud Credentials Locally
You need this for DVC to get and upload data from our data bucket on GCP. Follow these instructions.

## Get Latest Data
If your dataset folder is empty, that's a good sign for you to pull the latest data that this project depends on. You'll have to install DVC first, after that simply run:<br>
`dvc pull` <br> and it'll fetch the latest data from our secure google cloud bucket to the `dataset/` folder. Any issues pulling data could be from not configuring your Google Cloud credentials as described earlier.

## Push Latest Data
As a rule of thumb, put all datafiles in `dataset/` folder with descriptive names. 
### If you have data files that you want to add to version control:
1. Run `dvc add /path/to/data_file`
<br>This step is similar to adding a code file to git. It creates a new 'dvc' file with your new data file as it's name. It just tracks metadata about the file, also adds the data file to `.gitignore` so it doesn't push to git. 

2. Add the previously mentioned two files to git. DVC should also return the git command to add the two new changes to git. Command will change as per your data file.
`git add dataset\trial_data.csv.dvc dataset\.gitignore` <br> followed by a git commit <br>
`git commit -m "Added {your data file desc}"`

3. Run `dvc config core.autostage true`
<br> This step just automatically stages your data changes but doesn't commit them. That's the next step.

3. Run `dvc push` to push this latest data to the already configured google cloud bucket. If you run into errors, you probably did not setup the google cloud credentials. Probably ask Aakash or Samanvya.

### If you want to modify existing dataset already tracked by DVC

Let's say you modified a dataset file (added more rows or features), simply follow these steps so your teammates can get all the latest data as well.

1. `dvc commit` <br> This commits all changes made to existing datasets already tracked by DVC. To track new datasets, check the previous subheading.
2. `git add dataset/*.dvc && git commit -m "Edited {data change details}` <br> This commits any DVC files that have been changed by you.
3. `dvc push` pushes data to cloud bucket.
4. `git push...` commit your changes to git pertaining to the DVC files
