import re 

def text_clean(text:str):
    return text.replace("，","").replace("？","").replace("。","").replace("！","").\
                            replace("《","").replace("》","").replace("、","").replace("：","").\
                            replace("“","").replace("”","").replace(",","")

def s2c(sen):
    c = []
    buf = []
    for char in sen:
        if u'\u4e00 ' <= char <= u'\u9fff':
            if buf:
                c.append(''.join(buf))
                buf = []
            c.append(char)
        else:
            buf.append(char)
    if buf:
        c.append(''.join(buf))
        buf = []
    return ' '.join(c)

def abbr_mg(char:str):
    mg = []
    buf = []
    for chrs in char.split():
        if len(chrs)==1 and re.findall("[0-9a-zA-Z]",chrs):
            buf.append(chrs)
        else:
            if buf:
                mg.append(''.join(buf))
                buf = []
            mg.append(chrs)
    if buf:
        mg.append(''.join(buf))
        buf = []
    return ' '.join(mg)



def min_edit_distance(a, b):
        dp = [[0 for i in range(len(b) + 1)] for j in range(len(a) + 1)]
        for i in range(len(a) + 1):
            dp[i][0] = i
        for j in range(len(b) + 1):
            dp[0][j] = j
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
        # print(dp[-1][-1])
        return dp[-1][-1]

def compute_min_edit_distance(text1,text2,show=True):
    char1 = abbr_mg(s2c(text_clean(text1)))
    char2 = abbr_mg(s2c(text_clean(text2)))
    if show:
        print("------------- ref ---------------")
        print(char1)
        print("------------- hyp ---------------")
        print(char2)
    med = min_edit_distance(char1.split(),char2.split())
    return med, len(char1.split())

def compute_mer_text(text1,text2,show=False):
    char1 = abbr_mg(s2c(text_clean(text1)))
    char2 = abbr_mg(s2c(text_clean(text2)))
    if show:
        print("------------- ref ---------------")
        print(char1)
        print("------------- hyp ---------------")
        print(char2)
    mec = min_edit_distance(char1.split(),char2.split())
    return mec/len(char1.split())


def compute_mer_file(file1,file2,show=False):
    with open(file1,"rt",encoding="utf-8") as f:
        lines1 = f.readlines()
    with open(file2,"rt",encoding="utf-8") as f:
        lines2 = f.readlines()

    text1 = " ".join(lines1)
    text2 = " ".join(lines2)
    return compute_mer_text(text1,text2,show)

    

if __name__ == "__main__":

    import sys 
    res = compute_mer_file(sys.argv[1],sys.argv[2])
    print(res)