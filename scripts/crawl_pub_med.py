import urllib.request
import os


destination = '/home/pfb16181/all_PubMed_Data'
start = 1
end = 1016
uri = "ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
for i in range(start, end):
    num = "{0}".format(i)
    if len(num) < 4:
        p = 4 - len(num)
        pad = "0" * p

    url = "{0}pubmed20n{1}{2}.xml.gz".format(uri, pad, num)
    filename = "{0}/pubmed20n{1}{2}.xml.gz".format(destination, pad, num)
    urllib.request.urlretrieve(url, filename)
    try:
        print(url)
    except:
        print("Skipping " + filename)
