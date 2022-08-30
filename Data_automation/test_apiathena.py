from apiathena import apirequest 


myquery='''


select * from  main_db.year_country_totalemision limit 50;

 
'''

df = apirequest(myquery)