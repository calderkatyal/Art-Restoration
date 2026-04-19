import torch

from src.dataset import DistributedStatefulEpochSampler, StatefulEpochSampler


def test_stateful_epoch_sampler_resume_mid_epoch():
    sampler = StatefulEpochSampler(list(range(8)), shuffle=True, seed=17)
    iterator = iter(sampler)
    consumed = [next(iterator) for _ in range(3)]
    state = sampler.state_dict()

    resumed = StatefulEpochSampler(list(range(8)), shuffle=True, seed=17)
    resumed.load_state_dict(state)

    assert consumed == sampler.seen_indices()
    assert list(iter(resumed)) == sampler.remaining_indices()


def test_distributed_sampler_shards_match_padded_global_order():
    dataset = list(range(10))
    samplers = [
        DistributedStatefulEpochSampler(
            dataset,
            num_replicas=3,
            rank=rank,
            shuffle=True,
            seed=29,
            drop_last=False,
        )
        for rank in range(3)
    ]

    orders = [sampler.current_order() for sampler in samplers]
    global_order = []
    for position in range(len(orders[0])):
        for rank_order in orders:
            global_order.append(rank_order[position])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(29)
    shuffled = torch.randperm(len(dataset), generator=generator).tolist()
    expected = shuffled + shuffled[:2]

    assert all(len(order) == 4 for order in orders)
    assert global_order == expected


def test_distributed_sampler_resume_mid_epoch():
    sampler = DistributedStatefulEpochSampler(
        list(range(11)),
        num_replicas=2,
        rank=1,
        shuffle=True,
        seed=101,
        drop_last=False,
    )
    iterator = iter(sampler)
    consumed = [next(iterator) for _ in range(2)]
    state = sampler.state_dict()

    resumed = DistributedStatefulEpochSampler(
        list(range(11)),
        num_replicas=2,
        rank=1,
        shuffle=True,
        seed=101,
        drop_last=False,
    )
    resumed.load_state_dict(state)

    assert consumed == sampler.seen_indices()
    assert list(iter(resumed)) == sampler.remaining_indices()


def test_distributed_sampler_drop_last_matches_expected_lengths():
    samplers = [
        DistributedStatefulEpochSampler(
            list(range(10)),
            num_replicas=3,
            rank=rank,
            shuffle=False,
            seed=0,
            drop_last=True,
        )
        for rank in range(3)
    ]

    orders = [sampler.current_order() for sampler in samplers]
    flattened = []
    for position in range(len(orders[0])):
        for rank_order in orders:
            flattened.append(rank_order[position])

    assert all(len(order) == 3 for order in orders)
    assert flattened == list(range(9))


def test_distributed_sampler_set_epoch_changes_shuffle_order():
    sampler = DistributedStatefulEpochSampler(
        list(range(12)),
        num_replicas=2,
        rank=0,
        shuffle=True,
        seed=5,
        drop_last=False,
    )

    epoch_zero = sampler.current_order()
    sampler.set_epoch(1)
    epoch_one = sampler.current_order()

    assert epoch_zero != epoch_one
