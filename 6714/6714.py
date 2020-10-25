#
# @author: pj4dev.mit@gmail.com
# @desc: Positional Indexes Intersection
# @comment: written in Python 2.7.x
# @ref: http://nlp.stanford.edu/IR-book/html/htmledition/positional-indexes-1.html
#

def docID(plist):
    return plist[0]


def position(plist):
    return plist[1]


# p is a dollar list
# dollar = [[1, [8, 15, 20, 26, 34]], [4, [8, 15, 20, 26, 34, 500]], [5, [8, 15, 20, 26, 34, 400]]]
def skipTo(p, docID, pos):
    for item in p:
        if item[0] == docID:
            for position in item[1]:
                if position > pos:
                    return position


def pos_intersect(p1, p2, dollar):
    answer = []  # answer <- ()
    len1 = len(p1)
    len2 = len(p2)
    i = j = 0
    while i != len1 and j != len2:  # while (p1 != nil and p2 != nil)
        if docID(p1[i]) == docID(p2[j]):
            l = []  # l <- ()
            pp1 = position(p1[i])  # pp1 <- positions(p1)
            pp2 = position(p2[j])  # pp2 <- positions(p2)

            plen1 = len(pp1)
            plen2 = len(pp2)
            ii = jj = 0
            while ii != plen1:  # while (pp1 != nil)
                # 增加 ---- start ------


                # $$$$$$$$$$$$$$$$$$$$$$     pos(pp1) is pp1[ii]  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                start_position = 0
                end_position = skipTo(dollar, docID(p1[i]), pp1[ii])
                while skipTo(dollar, docID(p1[i]), start_position) < end_position:
                    start_position = skipTo(dollar, docID(p1[i]), start_position)


                # 增加 ---- end ------
                while jj != plen2:  # while (pp2 != nil)


                    # if abs(pp1[ii] - pp2[jj]) <= k:  # if (|pos(pp1) - pos(pp2)| <= k)
                    #     l.append(pp2[jj])  # l.add(pos(pp2))
                    # elif pp2[jj] > pp1[ii]:  # else if (pos(pp2) > pos(pp1))
                    #     break

                    # 修改 start
                    if pp2[jj] not in range(start_position, end_position):
                        if pp2[jj] > start_position:
                            break
                    else:
                        if pp2[jj] > pp1[ii]:
                            l.append(pp2[jj])  # l.add(pos(pp2))
                    jj += 1  # pp2 <- next(pp2)

                    # 修改 end


                while l != [] and l[0] < pp1[ii] :  # while (l != () and |l(0) - pos(pp1)| > k)
                    l.remove(l[0])  # delete(l[0])


                for ps in l:  # for each ps in l
                    answer.append([docID(p1[i]), pp1[ii], ps])  # add answer(docID(p1), pos(pp1), ps)
                ii += 1  # pp1 <- next(pp1)
            i += 1  # p1 <- next(p1)
            j += 1  # p2 <- next(p2)
        elif docID(p1[i]) < docID(p2[j]):  # else if (docID(p1) < docID(p2))
            i += 1  # p1 <- next(p1)
        else:
            j += 1  # p2 <- next(p2)
    return answer


def run_test():
    print("to be or not to be")
    to = [[1, [7, 15, 18, 33]], [2, [1, 17, 74, 222, 255]], [4, [8, 16, 190, 429, 433]], [5, [363, 367]],
          [7, [13, 23, 191]]]
    be = [[1, [17, 25]], [4, [17, 191, 291, 430, 434]], [5, [14, 19, 101]]]
    dollar = [[1, [8, 15, 20, 26, 34]], [4, [8, 15, 20, 26, 34, 500]], [5, [8, 15, 20, 26, 34, 400]]]
    print("to: ", to)
    print("be: ", be)
    print("dollar:", dollar)


    print("merge result of \"to /1 be\": ", pos_intersect(to, be, dollar))

if __name__ == '__main__':
    run_test()