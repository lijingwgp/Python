'''
    PDF Extraction
    Hospital Pharmacy Data
    Aaron Koehl

    ChangeLog:
        2017-AUG-16 Created. AK
'''

import re
import sys

f = open("c:/tmp/test.txt",'rt')

#fOut = open("c:/tmp/out.txt",'wt')
fOut = sys.stdout
fOut.write("Drug\tDate\tTime\ttVerify\ttAdmin\tPriority\tLoc\ttID\n")

r = re.compile("^(.*?)(\d{2}/\d{2}/\d{4})\s+(\d+)\s{0,15}?(.{0,20}(min)?)(.{0,40})\s*(.*)")  # TH and DHW

dec = re.compile('[^\d.-]+')
hdr = re.compile("^([A-Z]{3,4}[^\[]{25}?\s{15})")

header = ''
rowCount = 0
for l in f:
    l = l.strip()
    if len(l) == 0:
        continue
    m=hdr.match(l)
    if m:
        header = m.group(1)
        q = re.search("^(.*?)( HOSPITAL|\(cont'd\)|\s{3})",header) # strip HOSPITAL and (cont'd)
        header = q.group(1)
        # print header
        continue

    m = r.search(l)
    tNum = ''
    if m:
        cols = []
        for ncol in range(7):   # TODO: change to 8
            if m.group(ncol+1) is not None:
                cols.append(m.group(ncol+1).strip().replace("'","\\'"))
        cols.append(header)
        m = re.search("(.*?)\[(\d+)\]$",cols[0])
        if m:
            tNum = m.group(2)
            cols[0] = m.group(1)
            col0 = ''
    else:
        m = re.search("(.*?)\[(\d+)\]$",l)
        if m:
            col0 = ' ' + m.group(1).strip()
            tNum = m.group(2)

    if tNum != '':
        rowCount += 1
        cols.append(tNum)
        cols[3] = cols[3].strip(' mins')
        cols[4] = cols[4].strip(' mins')
        cols[5] = cols[5].strip(' mins')
        fOut.write('\t'.join(cols))
        fOut.write('\n')

print str(rowCount) + ' records.'