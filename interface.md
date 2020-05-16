



run_state = Runstate()

run_state.idle()



net = model(rt_state=run_state)

mapping = net.generate_mapping()

list = net.generate_list()
 


runtime = Runtime(mapping, run_state) #operate on the mapping and run_state



cat $(ls -td -- */ | head -n 1)output.log.0

 python driver.py --config_file driver_configs/bert_4pipedream.yml --launch_single_container



