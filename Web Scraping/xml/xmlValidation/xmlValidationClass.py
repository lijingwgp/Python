# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 11:36:23 2016

@author: james.bradley
"""

from sys import *
from lxml import etree, objectify
from StringIO import StringIO

error_found = False
xml_doc_path = './CourseScheduleInvalid.xml'
f = open(xml_doc_path)
xml_doc = f.read()
f.close()
f = open(xml_doc_path)
xml_doc_lines = f.readlines()
f.close()

#print xml_doc_lines

xml_doc_lines_strip = []
for this_line in xml_doc_lines:
    xml_doc_lines_strip.append(this_line.rstrip('\n'))
    
#print xml_doc_lines_strip

f = open('./COURSE_SCHEDULE.dtd')
dtd_doc = f.read()
f.close()

"""
print xml_doc
print 'len(): ' + str(len(xml_doc))
"""

dtd = etree.DTD(StringIO(dtd_doc))
tree = objectify.parse(StringIO(xml_doc))
dtd.validate(tree) 

"""
print xml_doc
print 'len(): ' + str(len(xml_doc))
print 'type(xml_doc): ' + str(type(xml_doc))
"""

"""
print xml_doc
print 'len(): ' + str(len(xml_doc))
print 'type(xml_doc): ' + str(type(xml_doc))
"""

"""
for this_line in range(len(xml_doc_lines)):
    print xml_doc_lines[this_line]
"""
    
#print 'len(tree): ' + str(tree.)
if len(dtd.error_log.filter_from_errors()) > 0:
    error_found = True
    for error in dtd.error_log.filter_from_errors():
        #print(error.message)
        #print(error.line)
        #print(error.column)
        error_message = error.message
        error_line = error.line
        error_column = error.column
    for this_line in range(len(xml_doc_lines_strip)):
        for this_char in range(len(xml_doc_lines_strip[this_line])):
            #print this_line, this_char
            if this_line == error_line and this_char == error_column:
                #print ('\033[0;30;41m ' + xml_doc_lines_strip[this_line][this_char])
                stdout.write('\033[0;30;41m' + xml_doc_lines_strip[this_line][this_char])
                 #print "Here it is"
            else:
                #print ('\033[0m ' +xml_doc_lines_strip[this_line][this_char])
                stdout.write('\033[0m' +xml_doc_lines_strip[this_line][this_char])
        print
    #print
    print "\n\033[0;31m Error Message: " + error_message
    print '\033[0;30m'
    #print
else:
    print "\n\nXML doc okay"
            
        
"""        
thistree = etree.parse(StringIO(xml_doc))
r = thistree.xpath('/COURSE_SCHEDULE/COURSE_NAME')
for rr in r:
    print rr.text
    
print

s = thistree.xpath('/COURSE_SCHEDULE/SESSION/SESSION_NUMBER')
for ss in s:
    print ss.text, type(ss)
    
print
print

s = thistree.xpath('/COURSE_SCHEDULE/SESSION')
print 'len(s): ',len(s)
for ss in s:
    print ss.tag, type(ss)
    try:
        tt = ss.getnext()
        print tt.tag
    except:
        continue
"""