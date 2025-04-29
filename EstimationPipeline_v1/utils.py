def update_slack_counter(master_pb, slack_counter):
    num_constrs_removed = 0
    for constr in master_pb.getConstrs():
        if constr.ConstrName not in slack_counter:
            slack_counter[constr.ConstrName] = 0
        if constr.Slack < 0:
            slack_counter[constr.ConstrName] += 1

        if slack_counter[constr.ConstrName] >= slack_counter["MAX_SLACK_COUNTER"]:
            master_pb.remove(constr)
            slack_counter.pop(constr.ConstrName)
            num_constrs_removed += 1

    return slack_counter, num_constrs_removed