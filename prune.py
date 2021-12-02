import torch


def freeze_pruned_weights(model: torch.nn.Module, epsilon: float):
    """
    Nuke the gradients for all the weights which are small enough.
    Ideally we could fork the optimizer and make it pruning aware
    but this is the poor man's take
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "weight" in name:
                tensor = p.data
                grad_tensor = p.grad
                grad_tensor = torch.where(
                    tensor.abs() < epsilon, torch.zeros_like(grad_tensor), grad_tensor
                )
                p.grad.data = grad_tensor


def rewind_model(
    model: torch.nn.Module, model_snapshot: torch.nn.Module, epsilon: float
):
    with torch.no_grad():
        for (name, p), (_, p_snap) in zip(
            model.named_parameters(), model_snapshot.named_parameters()
        ):
            if "weight" in name:
                # Only rewind the weights which are not frozen
                p.data = torch.where(p.data.abs() > epsilon, p_snap.data, p.data)
                print(p)

            else:
                # Completely rewind
                p.data = p_snap.data


def async_push_lottery_tickets(
    client_store,
    model: torch.nn.Module,
    max_pruning_per_layer: float,
):
    # Use the distributed e thore to send async the lottery tickets
    # store.set()
    pass


def async_get_lottery_tickets(
    client_store,
    model: torch.nn.Module,
    max_pruning_per_layer: float,
):
    # Use the distributed e thore to send async the lottery tickets
    # store.get()
    pass


def sync_exchange_lottery_tickets(
    rank: int,
    model: torch.nn.Module,
    epsilon: float,
    max_pruning_per_layer: float,
    vote_threshold: int,
):
    """
    Each agent prunes its weights, and exchanges the pruned coordinates with the others.
    """

    with torch.no_grad():
        overall_pruned = 0.0
        overall_parameters = 0

        for name, p in model.named_parameters():
            if "weight" in name:
                # Find the local weights which should be pruned
                local_prune = p.data.abs() < epsilon

                # Share that with everyone. all_reduce requires ints
                shared_prune = local_prune.to(torch.int32)

                torch.distributed.all_reduce(
                    shared_prune, op=torch.distributed.ReduceOp.SUM
                )

                # Only keep the pruning which is suggested by enough agents
                shared_prune = shared_prune > vote_threshold

                print(
                    rank,
                    f"{torch.sum(local_prune)} pruned locally, {torch.sum(shared_prune)} pruned collectively",
                )

                # Prune following the collective knowledge
                if torch.sum(shared_prune) / p.numel() < max_pruning_per_layer:
                    p.data = torch.where(shared_prune, torch.zeros_like(p.data), p.data)

                    # Bookkeeping:
                    overall_pruned += torch.sum(shared_prune).item()
                    overall_parameters += p.numel()

    pruning_ratio = overall_pruned / overall_parameters

    if rank == 0:
        print(f"Model is now {pruning_ratio:.2f} pruned")

    return pruning_ratio


def sync_exchange_lottery_tickets_sorted(
    rank: int,
    model: torch.nn.Module,
    desired_pruning_ratio: float,
):
    """
    Each agent prunes a fixed ratio of weights based on the shared amplitudes.

    FIXME: We could prune individually instead and share the nulls
    """
    pruned_parameters = 0
    total_parameters = 0

    with torch.no_grad():
        for name, p in model.named_parameters():
            if "weight" in name:
                # We consider the absolute values on purpose, so that
                # weights which have very different distributions across the fleet are not nuked
                amplitudes = p.data.detach().clone().abs()
                torch.distributed.all_reduce(
                    amplitudes, op=torch.distributed.ReduceOp.SUM
                )

                # Prune the smallest ones first
                isort = torch.argsort(amplitudes)
                i_max = int(desired_pruning_ratio * amplitudes.numel())

                pruned_parameters += i_max
                total_parameters += amplitudes.numel()

                p.data[isort[:i_max]] = 0.0

    if rank == 0:
        print(f"Model is now {pruned_parameters/total_parameters:.2f} pruned")
