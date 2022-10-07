from hstest import *


ANSWER = [0.0, 'Sex', 0, [0, 4, 5, 6, 7], [1, 2, 3, 8, 9]]


class SplitTest(StageTest):

    @dynamic_test
    def test(self):
        pr = TestedProgram()
        pr.start()
        if not pr.is_waiting_input():
            raise WrongAnswer("You program should input the path to the file")
        output = pr.execute("test/data_stage2.csv").strip()
        output = output.replace(",", "").replace("]", "[").split("[")
        output = [x for x in output if x != "" and x != " "]
        if len(output) != 3:
            raise WrongAnswer("Your answer should contain five items: float value, string, value and two lists.")
        res = []
        first = output[0].strip().split()
        try:
            res.append(float(first[0]))
        except Exception:
            raise WrongAnswer("The first value should be float.")
        res.append(first[1])
        try:
            res.append(int(first[2]))
        except Exception:
            raise WrongAnswer("The third value should be integer.")
        try:
            res.append([int(x) for x in output[1].split()])
        except Exception:
            raise WrongAnswer("The fourth item should be a list of integers.")
        try:
            res.append([int(x) for x in output[2].split()])
        except Exception:
            raise WrongAnswer("The fifth item should be a list of integers.")
        if len(res) != 5:
            raise WrongAnswer("Your answer should contain five items: float value, string, value, two lists.")
        if res[0] != ANSWER[0]:
            raise WrongAnswer("Wrong Gini score (the first item in your answer).")
        if res[1] != ANSWER[1]:
            raise WrongAnswer("Wrong threshold feature (the second item in your answer).")
        if res[2] != ANSWER[2]:
            raise WrongAnswer("Wrong threshold value (the third item in your answer).")
        if res[3] != ANSWER[3]:
            raise WrongAnswer("Wrong list of left node indexes (the fourth item in your answer).")
        if res[4] != ANSWER[4]:
            raise WrongAnswer("Wrong list of right node indexes (the fifth item in your answer).")
        return CheckResult.correct()


if __name__ == '__main__':
    SplitTest().run_tests()
