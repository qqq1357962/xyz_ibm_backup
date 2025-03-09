import os

root_url = "http://iccad-contest.org/Problems/testcase"

benchmarks = [
    'Problem%20B_case1_0506.txt', 'Problem%20B_case2_0516.txt', 
    'Problem%20B_case3_0516.txt', 'Problem%20B_case4_0524.txt'
]

def run(cmd):
    print(cmd)
    os.system(cmd)

run("rm -rf *.txt")
for idx, bm in enumerate(benchmarks):
    run("wget %s/%s" % (root_url, bm))
run("chmod -x *.txt")
