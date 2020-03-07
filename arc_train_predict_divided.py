for output_id in submission.index:
    task_id = output_id.split('_')[0]
    pair_id = int(output_id.split('_')[1])
    f = str(test_path / str(task_id + '.json'))
    with open(f, 'r') as read_file:
        task = json.load(read_file)
    train_data = task['train']
    test_data = np.array(task['test'][pair_id]['input']) # test pair input
    try:
        train_divided_in = []
        train_divided_out = []
        for i in range(len(train_data)):
            try:
                train_in = np.array(train_data[i]['input']) 
                train_out = np.array(train_data[i]['output'])
                train_divided_in.append(divider(train_in, train_out)[0])
                train_divided_out.append(divider(train_in, train_out)[1])
            except:
                pass

        train_divided_in = [y for x in train_divided_in for y in x]
        train_divided_out = [y for x in train_divided_out for y in x]
            
        test_shape = divider(train_in, train_out)[2]
            
        kernel_pairs = []
        kernel_signatures = []
        for i in range(len(train_divided_in)):
            kernel_pairs.append([train_divided_in[i], train_divided_out[i]])
            kernel_signatures.append([kernel_signature(train_divided_in[i]), 
                                          kernel_signature(train_divided_out[i]),
                                          sign_union(train_divided_in[i], train_divided_out[i])])
                
        voc_check = []
        for i in range(len(kernel_signatures)):
            value1 = kernel_signatures[i][0]
            value2 = kernel_signatures[i][1]
        for j in range(len(kernel_signatures)):
            if kernel_signatures[j][0] == value1 and kernel_signatures[j][1] != value2:
                voc_check.append(j)
            
        voc_check = unique(voc_check)
        abc = alphabet_maker(voc_check, kernel_pairs)
            
        divided_test = test_divider(test_data)[0]
        if len(voc_check) != len(train_divided_in):
            for i in range(len(divided_test)):
                divided_test[i] = alphabet_modification(divided_test[i], abc)
                    
            #train_loop_check_list = []
            #for j in range(len(train_divided_in)):
            #    if j in voc_check:
            #        rows_list1 = rows_clip_equals(train_divided_in[j], train_divided_out[j])
            #        rows_list2 = rows_clip_nonequals(train_divided_in[j], train_divided_out[j])
            #        cols_list1 = cols_clip_equals(train_divided_in[j], train_divided_out[j])
            #        cols_list2 = cols_clip_nonequals(train_divided_in[j], train_divided_out[j])
                    
            #        train_loop_check_list.append(rows_list1)
            #        train_loop_check_list.append(rows_list2)
            #        train_loop_check_list.append(cols_list1)
            #        train_loop_check_list.append(cols_list2)
        
            #        divided_test[i] = rows_handler(rows_list1, divided_test[i])
            #        divided_test[i] = rows_handler(rows_list2, divided_test[i])
            #        divided_test[i] = cols_handler(cols_list1, divided_test[i])
            #        divided_test[i] = cols_handler(cols_list2, divided_test[i])
                    
        pred_1 = connector(divided_test, test_shape)
        pred_1 = flattener(pred_1.tolist())
        pred_2 = pred_1
        pred_3 = pred_2
            
    except:
        pred_1 = flattener(np.zeros((test_data.shape[0], test_data.shape[1]), dtype=int).tolist())
        pred_2 = flattener(np.zeros((test_data.shape[0], test_data.shape[1]), dtype=int).tolist())
        pred_3 = flattener(np.zeros((test_data.shape[0], test_data.shape[1]), dtype=int).tolist())
        
    pred = pred_1 + ' ' + pred_2 + ' ' + pred_3 + ' ' 
    submission.loc[output_id, 'output'] = pred
    
submission.to_csv('submission.csv')
