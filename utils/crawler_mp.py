import os, json, urllib
import requests as req
from multiprocessing import Pool
from operator import itemgetter

# Return biosamples which has these 6 HMs
def extractAllEntries():
    matrix = getRespon(base_url+chip_seq_matrix_url)
    for idx in range(5):
        if category == matrix["matrix"]["x"]["biosample_ontology.classification"]["buckets"][idx]["key"]:
            x_axis_labels = matrix["matrix"]["x"]["biosample_ontology.classification"]["buckets"][idx]["biosample_ontology.term_name"]["buckets"]
            entries = {x_axis_label["key"] for x_axis_label in x_axis_labels}
            break

    print("Start...")

    del_entries = set()
    for entry in entries:
        for HM in HMs:
            response = req.get(base_url+serach_url.format(HM, entry, category_url))
            if response.status_code != req.codes.ok:
                del_entries.add(entry)
                break
    else:
        for entry in del_entries:
            entries.remove(entry)
    
    print("==============Extracting {} entries finished==============".format(len(entries)))
    return entries

# Return respnse dictionary
def getRespon(url):
    response = req.get(url, headers={"accept": "application/json"})
    # Check status
    if response.status_code == req.codes.ok:
        return response.json()
    else:
        raise RuntimeError("Can't get response")

# Return the fewest warnings dataset
def warningFilter(search_results):
    experiment_audit_pair = []
    audit_type = ["ERROR", "NOT_COMPLIANT", "WARNING"]
    for search_result in search_results["@graph"]:
        experiment_id = search_result["@id"]
        audits = search_result["audit"].keys()
        audit_amount_pair = [experiment_id, 0, 0, 0]

        for idx, audit in enumerate(audit_type, start=1):
            try:
                audit_amount_pair[idx] = len(search_result["audit"][audit])
            except:
                continue
        experiment_audit_pair.append(audit_amount_pair)
    return sorted(experiment_audit_pair, key=itemgetter(1,2,3))[0][0]

# Return specified signal track
def formatFilter(experiment_datasets):
    signal_tracks = []
    for experiment_dataset in experiment_datasets["files"]:
        if (experiment_dataset["file_type"] == file_type) and (experiment_dataset["assembly"] != "hg19") and (experiment_dataset["analyses"][0]["status"] == "released"):
            default = 1 if ("preferred_default" in list(experiment_dataset.keys())) else 0 # Default analysis signal in individual dataset         
            signal_tracks.append([experiment_dataset["@id"], default, len(experiment_dataset["biological_replicates"])])
    return sorted(signal_tracks, key=itemgetter(1,2))[-1][0]

def createFolder(entry):
    path = dataset_dir + entry
    os.mkdir(path)
    for HM in HMs:
        os.mkdir(path + "/" + HM)

def download(entry, HM, signal_url):
    # print("--- {} loading".format(HM))
    response = req.get(base_url+signal_url+"@@download/{}.{}".format(signal_url[7:-1], file_type.split(" ")[0]), headers={"accept": "application/json"})
    if response.status_code == req.codes.ok:
        with open(dataset_dir + "{}/{}/{}.{}".format(entry, HM, signal_url[7:-1], file_type.split(" ")[0]), "wb") as f:
            f.write(response.content)
            print("--- {} finished".format(HM))

def main(entries):
    for idx, entry in enumerate(entries, start=1):
        pool = Pool()
        print("Begin to download {} ({}/{})".format(entry, idx, len(entries)))
        for HM in HMs:
            search_results = getRespon(base_url+serach_url.format(HM, entry, category_url))
            experiment_url = warningFilter(search_results)

            experiment_datasets = getRespon(base_url+experiment_url)
            signal_url = formatFilter(experiment_datasets)
    
            if not os.path.isdir(dataset_dir + entry):
                createFolder(entry)

            pool.apply_async(download, args=(entry, HM, signal_url))
            # Change formatFilter and check whether the downloaded file is correct
            # if not os.path.isfile(dataset_dir + "{}/{}/{}.{}".format(entry, HM, signal_url[7:-1], file_type.split(" ")[0])):
            #     removed_path = "dataset/{}/{}/{}/".format(category, entry, HM)
            #     removed_path += os.listdir(removed_path)[0]
            #     os.remove(removed_path)

            #     print("--- {} changed : {}".format(HM, signal_url[7:-1]+".bigWig"))
            #     pool.apply_async(download, args=(category, entry, HM, signal_url))
        
        pool.close()
        pool.join()
        print("===================================")

if __name__ == "__main__":
    file_type = "bigBed narrowPeak"
    category, category_url = "new", "cell%20line"  # Folder name, Website address
    HMs = ["H3K4me3", "H3K27ac", "H3K4me1", "H3K36me3", "H3K9me3", "H3K27me3"]

    base_url = "https://www.encodeproject.org"
    chip_seq_matrix_url = "/chip-seq-matrix/?type=Experiment&replicates.library.biosample.donor.organism.scientific_name=Homo%20sapiens&assay_title=Histone%20ChIP-seq&assay_title=Mint-ChIP-seq&status=released"
    serach_url = "/search/?type=Experiment&status=released&target.label={}&assay_title=Histone%20ChIP-seq&assay_title=Mint-ChIP-seq&biosample_ontology.term_name={}&replicates.library.biosample.donor.organism.scientific_name=Homo%20sapiens&biosample_ontology.classification={}"

    dataset_dir = "../dataset/{}/".format(category)
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    # entries = extractAllEntries()
    entries = ["H1"]
    main(entries)