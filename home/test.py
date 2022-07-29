import openpyxl, csv
 
wb = openpyxl.load_workbook('/Users/gkeol/Documents/datasets.xlsx')
ws = wb.active
 
start_row = 시작 열
end_row = 마지막 열
 
mylist = []
count = 1
for column in ws['B']:
     if count >= start_row and count <= end_row:
         mylist.append(column.value)
     count += 1
 
with open("파일명.csv", "w") as f:
    wr = csv.writer(f, delimiter='\n')
    wr.writerow(mylist)