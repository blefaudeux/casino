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

            else:
                # Completely rewind
                p.data = p_snap.data


def exchange_lottery_tickets(
    rank: int,
    model: torch.nn.Module,
    epsilon: float,
    max_pruning_per_layer: float,
    vote_threshold: int,
):
    """
    Each agent prunes its weights, and exchanges the pruned coordinates with the others
    """

    with torch.no_grad():
        overall_pruned = 0
        overall_parameters = 0

        for name, p in model.named_parameters():
            if "weight" in name:
                # Find the local weights which should be pruned
                local_prune = p.data < epsilon

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
                    overall_pruned += torch.sum(shared_prune)
                    overall_parameters += p.numel()

    pruning_ratio = overall_pruned / overall_parameters

    if rank == 0:
        print(f"Model is now {pruning_ratio:.2f} pruned")

    return pruning_ratio