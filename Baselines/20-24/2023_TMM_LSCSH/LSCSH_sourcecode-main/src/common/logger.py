import datetime
import numpy

class Logger:

    @staticmethod
    def divider(title):
        print(
            "\n============================================{}=============================================\n".format(
                title),flush=True)

    @staticmethod
    def info(message):
        time = datetime.datetime.now()
        timeStr = time.strftime("[%Y%m%d-%H:%M:%S]")
        print(timeStr + '=>[info]: {}'.format(message),flush=True)

    @staticmethod
    def getTimeStr(time):
        return time.strftime("%Y-%m-%d_%H-%M-%S")


if __name__ == '__main__':
    a = numpy.arange(1, 10, 1)
    Logger.info("test {}".format(a))
    time = datetime.datetime.now()
    print(Logger.getTimeStr(time))
