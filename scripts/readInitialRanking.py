# author = Alexandros Ioannidis


def readInitialRanking(filename):
    list_of_pmids_for_topic = []
    record = False
    with open(filename, "r") as f:

        while f:

            line = f.readline()

            if not line:
                break

            if record == True:
                list_of_pmids_for_topic.append(line.strip())

            if line.startswith("Pids:"):
                record = True

    return(list_of_pmids_for_topic)
