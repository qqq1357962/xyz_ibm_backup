import os

root_url = "http://iccad-contest.org/Document/Problems/Testcase"
benchmarks = [
    'ProblemB_case1_0522.txt', 'ProblemB_case2.txt', 
    'ProblemB_case3_20230816.txt', 'ProblemB_case4_20230816.txt'
]
def run(cmd):
    print(cmd)
    os.system(cmd)

run("rm -rf *.txt")
for idx, bm in enumerate(benchmarks):
    run("wget %s/%s" % (root_url, bm))
run("chmod -x *.txt")
