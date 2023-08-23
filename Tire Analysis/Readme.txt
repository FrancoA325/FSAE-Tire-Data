BIG UPDATE

Doing some real documentaion now, The tire analysis file now does the longitudinal data fitting of the pacejka model. the tire file also now outputs excel files
that have pngs of the matplot lib graphs and all of the fitted data that the code outputs so you can compare tires better. 

to start, go to the fsaettc website and get the .mat files for the tires you want to analyize, make sure to get the cornering data and the drivebrake files because the code needs both to run, take the .mat file and import it in the python script go to the part where files are imported and change the file path to your tire files at the top of the code for cornering data and closer to the bottom for the drive brake data. also change the excel file output path and name to match the tire.

to scrape data at different IA or pressure, uncomment the code block in the two big loops or make changes to the code to what ever data you want to pull. make sure you understand the formatting of the runs and all of the imported and output files are named correctly.

take note that the data it outputs has a .65 scaling factor on all of the forces and tire friction coefficients as testing surface has better grip than the real world

if you have any questions send me an email at my school email fantunez@uncc.edu and i'll try to help.
I am still working on the format and small things. I may make changes in the future so there is no need to use matlab but i havent started to look at that yet.

Data used in this program is from Formula SAE Tire Test Consortium (FSAE TTC) and Calspan Tire Research Facility (TIRF)

AS OF 8/23/2023
The tire processing file now has a gui and there is no need to change the file paths directly in the code.

Data used in this program is from Formula SAE Tire Test Consortium (FSAE TTC) and Calspan Tire Research Facility (TIRF)
