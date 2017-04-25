id_deb = 58406
id_fin = 58743
with open("/home/nicolas/Documents/data_level_design/download", 'w') as file:
	for i in range(id_deb, id_fin + 1):
		file.write(" wget -O level_design_" + str(i) + 
			".jpg 'http://www.level-design.org/referencedb/action.php?id=" 
			+ str(i) + "&part=e&download' \n")
