import urllib.request

def stockurlgene(stockname):
	baseurl='https://www.alphavantage.co/query?'
	function='function=TIME_SERIES_DAILY_ADJUSTED'
	stock='&symbol='+stockname
	api='&apikey=GH8XQCKTN15SGYJJ'
	datatype='&datatype=csv'
	size='&outputsize=full'
	urlpost=baseurl+function+stock+api+datatype+size
	return(urlpost)
	
	

	
	
		








def CSVgene(filename,myurl):
	myreq = urllib.request.urlopen(myurl)    
	mydata = myreq.read()
	with open(filename, 'wb') as ofile:
		ofile.write(mydata)
	


CSVgene( 'GSPC.csv',stockurlgene("GSPC"))
CSVgene('IXIC.csv',stockurlgene("IXIC"))

CSVgene('Gold.csv',stockurlgene("GC=F"))


