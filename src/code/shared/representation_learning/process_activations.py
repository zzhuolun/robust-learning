import os
import argparse


def main(args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--filepath", type=str, help="path to activations file")
  parser = parser.parse_args(args)
  filepath = parser.filepath
  
  if filepath is None or (not os.path.exists(filepath)):
    return 0

  data = {}
  with open(filepath) as fp:
    lines = fp.readlines()
    for line in lines:
      epoch, batch, layer, activation = line.split(', ')
      epoch = int(epoch)
      layer = int(layer)
      activation = round(float(activation), 2)

      data[epoch] = {} if not epoch in data.keys() else data[epoch]
      data[epoch][layer] = [] if (not layer in data[epoch].keys()) else data[epoch][layer]
      data[epoch][layer].append(activation)


  fp = open(filepath, 'a+')
  fp.write("\n\n\n\nThe mean per epoch per layer activations are as follows:\n\n")
  for epoch, dt in data.items():
    mean_acts = []
    for layer, acts in dt.items():
      mean_acts.append( str( round( ( sum(acts) / float(len(acts)) ), 2) )  )
    print(epoch, ", ".join(mean_acts))
    fp.write("{}, {}\n".format( epoch, ", ".join(mean_acts)))
  fp.close()  

  return 1

if __name__ == '__main__':
  main()
