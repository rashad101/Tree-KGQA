from utils.dataset import Dataset
from components.entity_linking import EntityLinker
from components.relation_linking import RelationLinker
from components.kgqa import KGQA
from utils.KGQA_Exception import UnknownTaskException
from utils.args import get_args


def main(args):
    data = Dataset(args=args, dataset_name=args.dataset)

    if args.task == "EL":                           # entity linking
        el = EntityLinker(data=data, args=args)
        el.perform_EL()                             # perform entity linking over all the data
        el.save_predictions()

        if args.evaluate:
            el.evaluate()                           # evaluate and show the scores on standard metric

    elif args.task == "RL":
        rl = RelationLinker(data=data, args=args)
        rl.perform_RL_single(data[0],args)

    elif args.task == "KGQA":                       # KGQA task
        el = EntityLinker(data=data, args=args)     # initializes the entity linker for performing KGQA
        qa = KGQA(entity_linker=el, data=data, args=args)
        qa.perform_KGQA()
        qa.save_predictions()                       # evaluate and show scores on standard metric

        if args.evaluate:
            qa.evaluate()                           # evaluate and show the scores on standard metric
    else:
        raise UnknownTaskException(task=args.task)


if __name__ == "__main__":
    args = get_args()
    main(args)