'''
More info on pytesseract here:
https://pypi.python.org/pypi/pytesseract

Note that the package requires:
PIL/Pillow
Google Tesseract OCR (Windows installer can be found here: https://github.com/UB-Mannheim/tesseract/wiki)
'''
import numpy as np
import csv
import collections
import re
import os
import os.path
import time
import sys

timeStarted = time.clock()

if len(sys.argv) != 6:
    print('Usage:\npython {0} filepath.png xCells yCells cellProp borderWidth'.format(sys.argv[0]))
    print('Where:\n'
          '  cellProp refers to the proportion of the cell that is taken up by the number at the top of each cell\n'
          '  borderWidth refers to the width of the border between cells, which should be adjusted for (usually ~0.1)')
    print('\ne.g. for a standard 13 x 13 codeword where the number takes up the top 35% of a cell:\n"python {0} cw02.png 13 13 0.35 0.1"'.format(sys.argv[0]))
    exit()


# import opencv just to display the current image
import cv2

try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract
# need to specify tesseract path as not in environment variables
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'

# check that the 'cells' folder is there
if not os.path.exists("cells"):
    os.makedirs("cells")

# remove images from other runs
def remove_images():
    try:
        os.remove("cells\\cell.png")
        os.remove("cells\\number.png")
        os.remove("cells\\letter.png")
    except OSError:
        pass


def read_image(x, y):
    # get current cell and output to cell.png
    borderAdjustx = x * float(sys.argv[5])
    borderAdjusty = y * float(sys.argv[5])
    x_raw = int(round(x * cellWidth - cellWidth - borderAdjustx, 0))
    x_raw_end = int(round(x_raw + cellWidth, 0))
    y_raw = int(round(y * cellHeight - cellHeight - borderAdjusty, 0 ))
    y_raw_end = int(round(y_raw + cellHeight, 0))
    print("Processing ({}, {}), ({}:{}, {}:{})...\r".format(x, y, x_raw, x_raw_end, y_raw, y_raw_end), end = '')
    crop_img = img[y_raw:y_raw_end, x_raw:x_raw_end]
    cv2.imwrite("cells\\cell.png", crop_img)
    cell_w_standard = round(cellWidth / 50)
    cell_h_standard = round(cellHeight / 50)

    # draw rectangle around current cell in 'Codeword' window
    img_disp = cv2.imread(sys.argv[1])
    cv2.rectangle(img_disp, (x_raw, y_raw), (x_raw_end, y_raw_end), (0,255,0), 3)
    img_disp = cv2.resize(img_disp, (400, 400))
    cv2.imshow('Codeword', img_disp)

    # get number
    nr_width = int(round(cellWidth / 1.5))
    nr_height = int(round(cellHeight * float(sys.argv[4])))
    number = crop_img[2*cell_h_standard:nr_height, round(2.5*cell_w_standard):nr_width]
    number = cv2.resize(number, (0, 0), fx=20, fy=20)
    cv2.imwrite("cells\\number.png", number)
    cv2.imshow('Number', number)
    number_read = pytesseract.image_to_string(Image.open('cells\\number.png'))
    number_read = re.sub("[^0-9]", "", number_read)
    if number_read == "":
        number_read = pytesseract.image_to_string(Image.open('cells\\number.png'), config='-psm 6')
        number_read = re.sub("[^0-9]", "", number_read)

    # get letter
    letter = crop_img[nr_height:round(cellHeight)-5*cell_w_standard, 10*cell_w_standard:round(cellWidth)-10*cell_w_standard]
    coord = [(x * cellWidth - cellWidth), (y * cellHeight)]
    letter = cv2.resize(letter, (0, 0), fx=10, fy=10)
    cv2.imwrite("cells\\letter.png", letter)
    cv2.imshow('Letter', letter)
    letter_read = pytesseract.image_to_string(Image.open('cells\\letter.png'), config='-psm 6')
    letter_read = re.sub(r"[\W_\d]+", "", letter_read)

    if number_read == '':
        number_print = '-'
    else:
        number_print = number_read
    if letter_read == '':
        letter_print = '-'
    else:
        letter_print = letter_read
    print("                                                                   \r", end = '')
    print("({}, {}), ({}:{}, {}:{}): [{}, {}]".format(x, y, x_raw, x_raw_end, y_raw, y_raw_end, number_print, letter_print))
    cv2.waitKey(1)
    return number_read, letter_read, coord

remove_images()

# display file info
print("File:\t{}".format(sys.argv[1]))
img = cv2.imread(sys.argv[1], 0)
img_disp = cv2.imread(sys.argv[1])
img_disp = cv2.resize(img_disp, (400, 400))
cv2.imshow('Codeword', img_disp)
height, width = img.shape[:2]
print("Size:\t{} * {}".format(width, height))
print("x:\t{}".format(sys.argv[2]))
print("y:\t{}".format(sys.argv[3]))
print("cellProp:\t{}".format(sys.argv[4]))
print("borderWidth:\t{}".format(sys.argv[5]))
cellWidth = width / int(sys.argv[2])
cellHeight = height / int(sys.argv[3])
print("Cell size:\t{} * {}".format(round(cellWidth, 3), round(cellHeight, 3)))
print('')

table = []
print("Scanning image:")
for each_y in range(1, int(sys.argv[2]) + 1):
    row_data = []
    for each_x in range(1, int(sys.argv[3]) + 1):
        cellData = read_image(each_x, each_y)
        row_data.append(cellData)
    table.append(row_data)

remove_images()
# destroy number and letter windows
cv2.destroyWindow('Number')
cv2.destroyWindow('Letter')
# update Codeword window to get rid of last square
img_disp = cv2.imread(sys.argv[1])
img_disp = cv2.resize(img_disp, (400, 400))
cv2.imshow('Codeword', img_disp)
cv2.waitKey(1)

# transpose to get columns as lists and get words from columns
words_cols = []
table_trans = list(map(list, zip(*table)))
for row in table_trans:
    words_str = ''
    last_letter = ''
    for cell in row:
        if cell[0] == '':
            words_str = '{} '.format(words_str)
        else:
            if last_letter == '':
                words_str = '{}{}'.format(words_str, cell[0])
            else:
                words_str = '{},{}'.format(words_str, cell[0])
        last_letter = cell[0]
    words_cols.extend(words_str.split(' '))

# get words from rows
words_rows = []
for row in table:
    words_str = ''
    last_letter = ''
    for cell in row:
        if cell[0] == '':
            words_str = '{} '.format(words_str)
        else:
            if last_letter == '':
                words_str = '{}{}'.format(words_str, cell[0])
            else:
                words_str = '{},{}'.format(words_str, cell[0])
        last_letter = cell[0]
    words_rows.extend(words_str.split(' '))

# combine
words_raw = words_rows + words_cols
words = [item for item in words_raw if ',' in item]  # remove items with only 1 number in
print('\n Words detected:\n{}'.format(words))

# get known data
knownData = {}
knownDataReverse = {}
for row in table:
    for cell in row:
        if cell[0] != '' and cell[1] != '':
            if cell[1].lower() not in knownDataReverse:
                knownData[cell[0]] = cell[1].lower()
                knownDataReverse[cell[1].lower()] = cell[0]
print("\nKnown Data: {}".format(knownData))

# automatically skips the selection if more possible solutions than this:
# (if the number of possible solutions is less than this (but less than 30), will require input to disambiguate
# set to '1' for fully automated)
skip_if_more_than = 1

# location of zipfFreqs.csv
# (columns should be: Spelling, nchar, LogFreq_Zipf, DomPoS)
freqsCsvLoc='zipfFreqs.csv'
# prevents headers from being processed (e.g. prevents processing LogFreq_Zipf as float)
csvHasHeaders=True

# zipf data import
print('\nImporting word data from {0}...'.format(freqsCsvLoc))
try:
  reader = csv.reader(open(freqsCsvLoc))
  dataDict = {}
  csvIter=0
  for row in reader:
    if csvHasHeaders:
      if csvIter==0:
        pass
    if not csvHasHeaders or csvIter!=0:
        key = row[0].lower()
        if key in dataDict:
            # implement any duplicate row handling here. (May also want to do csvIter-=1 here)
          pass
        dataDict[key]=row[1:]
    csvIter+=1
  print(' -Done. Imported {0} entries.'.format(len(dataDict)))
except:
  print("Error importing from '{0}'. Check exists and formatted correctly.".format(freqsCsvLoc))


def define_target_word():
    global word_numbers_list
    word_numbers_list = []
    for each_word in words:
        word_as_list = each_word.split(',')
        word_numbers_list.append(word_as_list)


def find_matches():
    global reString
    reString = ''
    for nr in word_numbers:
        if nr in knownData:
            reString = ('{0}{1}'.format(reString, knownData[nr]))
        else:
            reString = ('{0}.'.format(reString))
    # where regular expression (re) treats '.' as a wildcard
    regex = re.compile(reString)
    global matches
    matches = [string for string in dataDict if re.match(regex, string)]
    # exclude words of wrong length
    matchesCorrLen = []
    for match in matches:
        if len(match) == len(reString):
            matchesCorrLen.append(match)
    matches = matchesCorrLen


def all_same(items):
    return all(x == items[0] for x in items)


def find_matches_with_repeats():  #e.g. in 355, the last two letters must be the same
    # count repeats
    global repeats
    repeats = {}
    for nr in word_numbers:
        if word_numbers.count(nr) > 1:
            repeats[nr] = [i for i, x in enumerate(word_numbers) if x == nr]
    # remove match if repeated letters don't match
    global matches
    matchesCorrReps = []
    for match in matches:
        cleaved = list(match)
        XS = []
        for key in repeats:
            cleaved_target_letters = []
            for item in repeats[key]:
                target_letter = cleaved[item]
                cleaved_target_letters.append(target_letter)
                if all_same(cleaved_target_letters):
                    XS.append('S')
                else:
                    XS.append('X')
        if all(outputs == 'S' for outputs in XS):
            matchesCorrReps.append(match)
    matches = matchesCorrReps


def exclude_wrong_repeats():
    global matches
    matchesCorrReps2 = []
    repeats_values = []
    for part in repeats.values():
        repeats_values.append(part)

    # count repeats in the word
    for match in matches:
        lttr_repeats = {}
        letters = list(match)
        for lttr in letters:
            if letters.count(lttr) > 1:
                indices = [i for i, x in enumerate(letters) if x == lttr]
                lttr_repeats[lttr] = indices

        lttr_repeats_values = []
        for part in lttr_repeats.values():
            lttr_repeats_values.append(part)
        if len(lttr_repeats) == len(repeats):  # number of letters repeated should match if correct
            if lttr_repeats_values == repeats_values:  # dictionary values should match if correct
                matchesCorrReps2.append(match)
    matches = matchesCorrReps2


def exclude_reusing_letters():  #e.g. if 7 is 's' then 7,2,12,7,2 cannot be 'sassa'
    global matches
    matchesReusedExcluded = matches
    for match in matches:
        lttr_iter = 0
        for lttr in match:
            if match in matchesReusedExcluded:
                if lttr in knownDataReverse:
                    if knownDataReverse[lttr] != word_numbers[lttr_iter]:
                        matchesReusedExcluded.remove(match)
            lttr_iter += 1
    matches = matchesReusedExcluded

def exclude_based_on_knownData():
    global matchesKDExcluded
    matchesKDExcluded = matches
    if len(matches) < 50:
        iterMax = len(matches)
    else:
        iterMax = 1
    for excludeIter in range(0, iterMax):
        for match in matches:
            for match_iter in range(0, len(word_numbers)):
                match_lttr = match[match_iter]
                target_nr = word_numbers[match_iter]
                if match_lttr in knownDataReverse:
                    if str(knownDataReverse[match_lttr]) != str(target_nr):
                        matchesKDExcluded.remove(str(match))
                        break


def getKey(item):
    return(item[1])


def getKey2(item):
    return(int(item[0]))


def sort_by_zipf():
    global matches
    matchesZipfs = []
    for match in matches:
        matchesZipfs.append([match, dataDict[match][1:2]])
    matchesZipfs.sort(key=getKey, reverse=True)
    matches = matchesZipfs


def print_results():
    global chosen
    skip = False
    if len(matches) > skip_if_more_than:
        print("{0} possible solutions found for '{1}'".format(len(matches), reString.replace('.', '_')))
        chosen = 0
        skip = True
    if len(matches) == 1:
        print("1 possible solution found for '{}'".format(reString.replace('.', '_')))
        chosen = 1
    else:
        if not skip:
            if len(matches) == 0:
                print("WORD NOT FOUND FOR '{}'".format(reString.replace('.', '_')))
                chosen = 0
            else:
                print('{} possible solutions found, ordered by likeliness:'.format(len(matches)))
                for solution in matches:
                    print(' {0}: {1}'.format(matches.index(solution) + 1, solution[0]))
                print(' 0: <skip> (if unsure)')
                chosen = input('\nChoose solution index: ')


already_done = False
define_target_word()
print('')
last_list_len = 0
lllDict = {}
lllDict2 = {}
startingLen = len(word_numbers_list)
while len(knownData) < 26:
    for wordNumberItem in word_numbers_list:

        # update cv2 codeword image
        img_disp = cv2.imread(sys.argv[1])
        img_disp = cv2.resize(img_disp, (400, 400))
        for row in table:
            for cell in row:
                if cell[0] != "" and cell[1] == "":
                    cellWidth400 = 400 / int(sys.argv[2])

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontSize = cellWidth400 / 120
                    if cell[0] in knownData:
                        text = knownData[cell[0]].upper()
                    else:
                        text = ''
                    textsize = cv2.getTextSize(text, font, 1, 2)[0]

                    xPos = round(((cell[2][0] / height) * 400 + cellWidth400 / 2) - (
                    textsize[0] / 4))  # ensures at centre of cell width-wise
                    yAdjust = round(cellWidth400 / 4.5)
                    yPos = round((cell[2][1] / width) * 400) - yAdjust

                    cv2.putText(img_disp, text, (xPos, yPos), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Codeword', img_disp)
        cv2.waitKey(1)

        # takes the numbers from the list and puts in a single string
        nr_text = ''
        for nr in wordNumberItem:
            if len(nr_text) == 0:
                nr_text = nr
            else:
                nr_text = '{0},{1}'.format(nr_text, nr)

        word_numbers = wordNumberItem
        print('\n{0}/{1} words solved   {2}/26 letters known\r'.format(startingLen - len(word_numbers_list), startingLen, len(knownData)))
        print('Processing word: {0}...'.format(nr_text))
        find_matches()
        find_matches_with_repeats()
        exclude_wrong_repeats()
        exclude_reusing_letters()
        exclude_based_on_knownData()
        matches = matchesKDExcluded
        sort_by_zipf()

        print_results()

        if chosen != 0 and chosen != '':
            chosen = int(chosen)
            chosenWord = matches[chosen - 1][0]
            print(' Word is {}'.format(chosenWord))
            # updates known info
            lttr_iter = 0
            for lttr in chosenWord:
                if lttr not in knownDataReverse:
                    matching_Nr = word_numbers[lttr_iter]
                    knownData[matching_Nr] = lttr
                    knownDataReverse[lttr] = matching_Nr
                    print(' {1} is {0}'.format(lttr, matching_Nr))
                lttr_iter += 1
            # removes from unsolved list
            word_numbers_list.remove(wordNumberItem)
        else:
            print('Skipped that selection')


        # if you know 25 letters, the 26th is that which is not defined
        if len(knownData) == 25:
            all_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
            for letter in all_letters:
                if letter not in knownDataReverse:
                    for number in range(0, 26):
                        if str(number+1) not in knownData:
                            knownData[str(number+1)] = letter
                            knownDataReverse[letter] = str(number+1)

        # if the length of the list of unsolved words is the same as 2 loops ago from the current word
        # then the codeword can't be solved
        if str(nr_text) in lllDict2:
            if int(len(word_numbers_list)) == lllDict2[nr_text]:
                print('\n\nCANNOT COMPLETE! SOLVED SO FAR:')
                totalSecs = time.clock() - timeStarted
                spareSecs = totalSecs % 60
                mins = int((totalSecs - spareSecs) / 60)
                print('(Took {0} mins, {1} secs)\n'.format(round(mins), round(spareSecs)))
                for itemNr in range(1, 27):
                    if len(str(itemNr)) < 2:
                        itemSpace = ' '
                    else:
                        itemSpace = ''
                    if str(itemNr) in knownData:
                        print('{0}{1} - {2}'.format(itemNr, itemSpace, knownData[str(itemNr)].upper()))
                    else:
                        print('{0}{1} - '.format(itemNr, itemSpace))
                already_done = True
                define_target_word()
                solvedNumbers = []
                solvedWords = []
                for item in word_numbers_list:
                    nr_text = ''
                    longest_nr_len = 0
                    for nr in item:
                        if len(nr_text) == 0:
                            nr_text = nr
                        else:
                            nr_text = '{0},{1}'.format(nr_text, nr)
                        if len(list(str(nr_text))) > longest_nr_len:
                            longest_nr_len = len(list(str(nr_text)))
                    solution_text = ''
                    for nr in item:
                        if len(solution_text) == 0:
                            if str(nr) in knownData:
                                solution_text = knownData[str(nr)]
                            else:
                                solution_text = '_'
                        else:
                            if str(nr) in knownData:
                                solution_text = '{0}{1}'.format(solution_text, knownData[str(nr)])
                            else:
                                solution_text = '{0}_'.format(solution_text)
                    solvedNumbers.append(nr_text)
                    solvedWords.append(solution_text.upper())
                print('')
                titles = ['Number', 'Word']
                dataToDisplay = [titles] + list(zip(solvedNumbers, solvedWords))
                for i, d in enumerate(dataToDisplay):
                    line = '| '.join(str(x).ljust(longest_nr_len*2) for x in d)
                    print(line)
                    if i == 0:
                        print('-' * len(line))
                exit()

        last_list_len = len(word_numbers_list)
        if str(nr_text) in lllDict:
            lllDict2[str(nr_text)] = lllDict[str(nr_text)]
        lllDict[str(nr_text)] = int(last_list_len)

        if len(knownData) >= 26:
            break

# finish cv2 codeword image
img_disp = cv2.imread(sys.argv[1])
img_disp = cv2.resize(img_disp, (400, 400))
for row in table:
    for cell in row:
        if cell[0] != "" and cell[1] == "":
            cellWidth400 = 400 / int(sys.argv[2])

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontSize = cellWidth400 / 120
            if cell[0] in knownData:
                text = knownData[cell[0]].upper()
            else:
                text = ''
            textsize = cv2.getTextSize(text, font, 1, 2)[0]

            xPos = round(((cell[2][0] / height) * 400 + cellWidth400 / 2) - (
                textsize[0] / 4))  # ensures at centre of cell width-wise
            yAdjust = round(cellWidth400 / 4.5)
            yPos = round((cell[2][1] / width) * 400) - yAdjust

            cv2.putText(img_disp, text, (xPos, yPos), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imshow('Codeword', img_disp)
cv2.waitKey(1)


# display solved info
print('\n\nSOLVED!')
totalSecs = time.clock() - timeStarted
spareSecs = totalSecs % 60
mins = int((totalSecs - spareSecs) / 60)
print('(Took {0} mins, {1} secs)\n'.format(round(mins), round(spareSecs)))
knownSummary = []
for key in knownData:
    knownSummary.append([key, knownData[key].upper()])
knownSummary.sort(key=getKey2)
for item in knownSummary:
    if len(str(item[0])) < 2:
        itemSpace = ' '
    else:
        itemSpace = ''
    print('{0}{1} - {2}'.format(item[0], itemSpace, item[1]))
already_done = True
define_target_word()
solvedNumbers = []
solvedWords = []
for item in word_numbers_list:
    nr_text = ''
    longest_nr_len = 0
    for nr in item:
        if len(nr_text) == 0:
            nr_text = nr
        else:
            nr_text = '{0},{1}'.format(nr_text, nr)
        if len(list(str(nr_text))) > longest_nr_len:
            longest_nr_len = len(list(str(nr_text)))
    solution_text = ''
    for nr in item:
        if len(solution_text) == 0:
            solution_text = knownData[nr]
        else:
            solution_text = '{0}{1}'.format(solution_text, knownData[nr])
    solvedNumbers.append(nr_text)
    solvedWords.append(solution_text.upper())
print('')
titles = ['Number', 'Word']
dataToDisplay = [titles] + list(zip(solvedNumbers, solvedWords))
for i, d in enumerate(dataToDisplay):
    line = '| '.join(str(x).ljust(longest_nr_len*2) for x in d)
    print(line)
    if i == 0:
        print('-' * len(line))

cv2.imshow('Codeword', img_disp)
print("\nPress any key on the Codeword window to exit...")
cv2.waitKey(0)

write_to_file = False
if write_to_file:
    cv2.waitKey(0)
    filePath = "out.png"
    img_disp = cv2.resize(img_disp, (width, height))
    cv2.imwrite(filePath, img_disp)
    print("\nWrote to '{}'".format(filePath))

# end message
end_message = 'Well done!!'.lower()
end_message_translated = ''
last_lttr = ' '
for lttr in end_message:
    if lttr == ' ':
        end_message_translated = '{} '.format(end_message_translated)
    else:
        if lttr in knownDataReverse:
            if last_lttr == ' ':
                end_message_translated = '{}{}'.format(end_message_translated, knownDataReverse[lttr])
            else:
                end_message_translated = '{},{}'.format(end_message_translated, knownDataReverse[lttr])
        if lttr not in knownDataReverse:
            end_message_translated = '{}{}'.format(end_message_translated, lttr)
    last_lttr = lttr
print('\n{}'.format(end_message_translated))
