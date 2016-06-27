from reader import read_file, convert_relations, read_file_Org, read_file_noSenses, convert_relations_noSenses
from readPBTD import subst_id_words
import json
import cPickle
import sys
import os

def test(inputfile, outputfile, parses_file):
    ''' run the reclassification '''

    #load trained neural network, label_subst, and semantic model (m)
    f1 = open("neuralnetwork_best.pickle", "rb")
    network = cPickle.load(f1)
    f1.close()
    f2 = open("label_subst_best.pickle", "rb")
    label_subst = cPickle.load(f2)
    f2.close()
    with open(parses_file) as p:
        parses = json.load(p)
    f4 = open("m_best.pickle", "rb")
    m = cPickle.load(f4)
    f4.close()
    inputfile_org = inputfile

    #check the TokenList type of the inputfile and convert, if necessary, with the subst_id_words method of readPDTB.py
    for row in open(inputfile_org, "r"):
       rel = json.loads(row)
       if len(rel["Arg1"])>0:
          try:
              token = rel["Arg1"]["TokenList"][0][0]
          except TypeError:
             print("Converting TokenList, please wait...")
             inputfile = subst_id_words(inputfile_org, parses_file) 
          except IndexError:
             print("Arg1 TokenList empty; checking the following relation...")
             continue
          break
    
    #get test_data
    print("Reading Data for Classification...")
    for row in open(inputfile, "r"):
       rel = json.loads(row)
       if len(rel["Sense"])== 0:#if file has no senses
          (relations_dev, all_relations_dev) = read_file_noSenses(inputfile, parses)
          (input_dev, output_dev) = convert_relations_noSenses(relations_dev, label_subst, m)
       else:
          if len(rel["Arg1"]["TokenList"][0]) == 2:#modified TokenList
              (relations_dev, all_relations_dev) = read_file(inputfile, parses)
          else:
              (relations_dev, all_relations_dev) = read_file_Org(inputfile, parses)
          (input_dev, output_dev) = convert_relations(relations_dev, label_subst, m)
       break    
    test_data = (input_dev, output_dev)
    
    #predict senses with network
    predicted_labels = network.predict(test_data[0])   
    lookup = dict([(y,x) for x,y in label_subst.items()])
    predicted_labels_names = [lookup[i] for i in predicted_labels]
    print("Length predicted_labels_names list: ", len(predicted_labels_names))

    #substitute senses
    f = open(inputfile_org, "r")#read from original inputfile (not modified, in case of TokenList Modification)
    filename = inputfile_org.split("/")[-1]
    g = open(outputfile, "w")
    new_rows = []
    counter = 0
    counter_lines = 0
    for row in f:
        counter_lines+=1
        new_row = ""
        rel = json.loads(row)

	#check TokenList Format and convert, if necessary
	try:
            token1 = rel["Arg1"]["TokenList"][0][0]
            rel["Arg1"]["TokenList"] = [i[2] for i in rel["Arg1"]["TokenList"]]
            try:
                token2 = rel["Arg2"]["TokenList"][0][0]
                rel["Arg2"]["TokenList"] = [i[2] for i in rel["Arg2"]["TokenList"]]
                try:
                    con = rel["Connective"]["TokenList"][0][0]
		    rel["Connective"]["TokenList"] = [i[2] for i in rel["Connective"]["TokenList"]]
                except TypeError:
                    pass
                except IndexError:
                    pass
            except TypeError:
                pass
            except IndexError:
                print("TokenList Arg2 empty")
                pass
        except TypeError:
            pass
        except IndexError:
            print("TokenList Arg1 empty")
            pass          

        #for relations-no-senses
        if rel["Type"] == '':#only look at the implicit relations
            if rel["Connective"]["TokenList"]==[]:
                if predicted_labels_names[counter] == u'EntRel':
                    rel["Type"] = "EntRel"
                else:
                    rel["Type"] = 'Implicit'#EntRel is also of Tpe 'Implicit'
            
                rel["Sense"] = [predicted_labels_names[counter]]
                new_row=json.dumps(rel)
                new_rows.append(new_row)
                counter+=1
        #for the other cases, where senses exist
        else:
            if rel["Type"] in ['Implicit', 'EntRel']:# and rel["Sense"]!=["EntRel"]:
                rel["Sense"] = [predicted_labels_names[counter]]
                new_row=json.dumps(rel)
                new_rows.append(new_row)
                counter+=1
            else:
                new_rows.append(json.dumps(rel))#explicit discourse relation 
    f.close()
    for row in new_rows:
        g.writelines(row+"\n")
    g.close()
    #delete modifiedTokenList file if exists
    file_to_delete = inputfile_org.split(".json")[0]+"_modifiedTokenList.json"
    if os.path.isfile(file_to_delete):
        os.remove(file_to_delete)


if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2], sys.argv[3])

#e.g. python re-classify-implicit_entrel_senses_nn.py input_to_be_reclassified_for_implicit_senses.json output_final.json data/en-01-12-16-dev/parses.json
