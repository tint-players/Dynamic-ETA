from __future__ import annotations

from backend.session import SessionManager
from simulator.ml_graph import (
    CROSSOVER_EDGE,
    ROUTE_SUCCESSOR_EDGE,
    MLGraphTopologyBuilder,
)


def _config():
    return SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)


def _operational_edges(builder: MLGraphTopologyBuilder):
    topology = builder.build()
    return [
        (
            topology.node_ids[src],
            topology.node_ids[dst],
            edge_type,
        )
        for src, dst, edge_type in zip(
            topology.operational_edge_index[0],
            topology.operational_edge_index[1],
            topology.operational_edge_types,
        )
    ]


def test_graph_topology_has_stable_16_node_canonical_order():
    topology = MLGraphTopologyBuilder(_config()).build()

    assert topology.node_ids == (
        "UP-BLK-01",
        "UP-BLK-02",
        "UP-BLK-03",
        "UP-BLK-04",
        "UP-BLK-05",
        "UP-BLK-06",
        "UP-BLK-07",
        "UP-BLK-08",
        "DOWN-BLK-01",
        "DOWN-BLK-02",
        "DOWN-BLK-03",
        "DOWN-BLK-04",
        "DOWN-BLK-05",
        "DOWN-BLK-06",
        "DOWN-BLK-07",
        "DOWN-BLK-08",
    )
    assert topology.num_nodes == 16


def test_operational_edges_follow_up_and_down_railway_directions():
    builder = MLGraphTopologyBuilder(_config())
    edges = _operational_edges(builder)

    for index in range(1, 8):
        assert (
            f"UP-BLK-{index:02d}",
            f"UP-BLK-{index + 1:02d}",
            ROUTE_SUCCESSOR_EDGE,
        ) in edges

    for index in range(8, 1, -1):
        assert (
            f"DOWN-BLK-{index:02d}",
            f"DOWN-BLK-{index - 1:02d}",
            ROUTE_SUCCESSOR_EDGE,
        ) in edges

    assert len([edge for edge in edges if edge[2] == ROUTE_SUCCESSOR_EDGE]) == 14


def test_agra_crossover_is_one_explicit_directed_operational_edge():
    builder = MLGraphTopologyBuilder(_config())
    edges = _operational_edges(builder)

    assert ("UP-BLK-07", "DOWN-BLK-07", CROSSOVER_EDGE) in edges
    assert ("DOWN-BLK-07", "UP-BLK-07", CROSSOVER_EDGE) not in edges
    assert len([edge for edge in edges if edge[2] == CROSSOVER_EDGE]) == 1


def test_message_passing_edge_index_is_bidirectional_without_changing_operational_truth():
    topology = MLGraphTopologyBuilder(_config()).build()
    message_edges = {
        (topology.node_ids[src], topology.node_ids[dst])
        for src, dst in zip(topology.edge_index[0], topology.edge_index[1])
    }

    assert topology.num_operational_edges == 15
    assert topology.num_message_edges == 30
    assert ("UP-BLK-01", "UP-BLK-02") in message_edges
    assert ("UP-BLK-02", "UP-BLK-01") in message_edges
    assert ("UP-BLK-07", "DOWN-BLK-07") in message_edges
    assert ("DOWN-BLK-07", "UP-BLK-07") in message_edges


def test_route_mask_is_configuration_derived_and_track_aware():
    config = _config()
    builder = MLGraphTopologyBuilder(config)

    primary_mask = builder.route_mask(config.train.train_id)
    primary_nodes = {
        node_id for node_id, included in zip(builder.node_ids, primary_mask) if included
    }
    assert {f"UP-BLK-{index:02d}" for index in range(1, 9)} <= primary_nodes
    assert "DOWN-BLK-07" in primary_nodes
    assert len(primary_nodes) == 9

    down_train_id = next(
        run.train.train_id
        for run in config.additional_train_runs
        if run.train.track_id == "TRACK-DOWN"
    )
    down_mask = builder.route_mask(down_train_id)
    down_nodes = {
        node_id for node_id, included in zip(builder.node_ids, down_mask) if included
    }
    assert down_nodes == {f"DOWN-BLK-{index:02d}" for index in range(1, 9)}


def test_route_mask_rejects_unknown_train():
    builder = MLGraphTopologyBuilder(_config())

    try:
        builder.route_mask("UNKNOWN-TRAIN")
    except KeyError as exc:
        assert "Unknown train_id" in str(exc)
    else:
        raise AssertionError("Expected unknown train to be rejected")
