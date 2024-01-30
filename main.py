# ============
# = ROCKMATE =
# ============

import rkgb
from rkgb.main import make_inputs
from rkgb.utils import print_debug, np, irotor
from rkgb.utils.global_vars import ref_verbose
from rkgb.utils.small_fcts import get_device
from rkgb.utils.ast_add_on import ast_to_str
from rockmate.def_op import DelOp, OpSchedule
from rockmate.def_chain import RK_Chain
from rockmate.def_sequence import (
    SeqBlockBwd,
    SeqBlockFc,
    SeqBlockFn,
    SeqBlockFe,
)
from rockmate.rotor_solver import seq_builder, solve_dp_functionnal
from rockmate.translator import Translator, RngState
from rockmate.compiler import Compiler, RK_Storage
import torch
from torch import tensor
import ast
import time
import pickle
from datetime import datetime
import warnings
from os import environ
from rockmate.def_op import *
from rockmate.def_chain import RK_Block

def print_memsizes(list_kg):
    di = list_kg[-1].dict_info
    for kg in list_kg:
        for n in kg.dict_nodes.values():
            mt = n.main_target
            try:
                print_debug(
                    f"{mt} : memsize {di[mt].memsize} ; " f"fm {n.fgt_mem}",
                    end="",
                )
            except:
                print_debug("\nloss")
    print_debug("\n")


class Rockmate(torch.nn.Module):
    def __init__(
        self,
        original_mod,
        model_inputs,
        budget=None,
        mem_unit=None,
        verbose=False,
        solve=True,
        get_sequence=True,
        get_compiled_fct=True,
        nb_budget_save=10,
        nb_budget_peak=5,
        ilp_solver="gurobi",
    ):
        super().__init__()
        ref_verbose[0] = verbose
        self.device = get_device()
        self.original_mod = original_mod
        self.mem_unit = mem_unit if mem_unit else 1024**2
        # -- use pytorch graph builder to get the list of K_graphs --
        self.rkgb_res = rkgb.make_all_graphs(
            original_mod, model_inputs, verbose=verbose, bool_kg=True
        )  # we don't need the whole K_graph
        
        
        self.dict_constants = self.rkgb_res.S_graph.dict_constants
        self.init_code = ast_to_str(self.rkgb_res.K_graph.init_code)
        self.output = self.rkgb_res.K_graph.output_kdn_data
        
        self.get_sequence2()
        self.get_compiled_fct()

    def get_chain(self, nb_budget_save=10, nb_budget_peak=5):
        start = time.time()
        #  -- use checkmate to solve all the blocks --
        self.rk_chain = RK_Chain(
            self.list_kg,
            self.eq_classes,
            nb_budget_save,
            nb_budget_peak,
            mem_unit=self.mem_unit,
        )
        end = time.time()
        self.ILP_solve_time = end - start

        self.opt_table = None

    def get_sequence2(self):
        def _fast_sched(kg):
            def _can_del(i, kdn):
                for kcn in kdn.users_real:
                    if kg.list_kcn.index(kcn) > i:
                        return False
                return True
            op_list = []
            alive_list = []
            alive_status = np.zeros(len(kg.list_kdn) + 2, dtype=bool)
            alive_status[-1] = True
            for i, kcn in enumerate(kg.list_kcn):
                op = RunOp(kcn)
                op_list.append(op)
                for kdn in kcn.users:
                    alive_status[kg.list_kdn.index(kdn)] = 1
                alive_list.append(alive_status.copy())
                for j, kdn in enumerate(kg.list_kdn):
                    if kdn in [kg.output_kdn_data, kg.output_kdn_grad]:
                        continue
                    if alive_status[j] and _can_del(i, kdn):
                        op = DelOp(kdn)
                        op.proxy = False
                        op_list.append(op)
                        alive_status[j] = 0
                        alive_list.append(alive_status.copy())
            return op_list, alive_list
        for n, p in self.original_mod.named_parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        kg = self.rkgb_res.K_graph
        op_list, alive_list = _fast_sched(kg)
        self.op_sched = OpSchedule(
            op_list,
            alive_list,
            kg.input_kdn_data,
            kg.input_kdn_grad,
            kg.output_kdn_data,
            kg.list_kdn
        )
        bwd_index_start = 0
        for i, op in enumerate(op_list):
            if 'bwd' in op.name:
                bwd_index_start = i
        self.fwd_op_list = op_list[:bwd_index_start]
        self.bwd_op_list = op_list[bwd_index_start:]
        self.storage = RK_Storage(self.device, self.original_mod, self.dict_constants)

    def get_compiled_fct(self):
        self.compiler = Compiler(self.storage)
        self.fct_list = self.compiler.compile(self.op_sched)
        loss_idx = len(self.fwd_op_list)
        self.fwd_fct_list = self.fct_list[:loss_idx]
        self.bwd_fct_list = self.fct_list[loss_idx:]

    def _exec(self, fct_list, record_mem=False, compiled=False):
        if not compiled:
            warnings.warn("Translator is no longer used!")
        if record_mem:
            torch.cuda.reset_peak_memory_stats()
            self.mem_before = torch.cuda.memory_allocated()
            self.max_before = torch.cuda.max_memory_allocated()
            for fct in fct_list:
                fct()
            allo_mem = torch.cuda.memory_allocated() - self.mem_before
            peak_mem = torch.cuda.max_memory_allocated() - self.max_before
            self.max_mem.append(peak_mem - allo_mem)
            self.allo_mem.append(allo_mem)
        else:
            for fct in fct_list:
                fct()

    def forward(self, *args, record_mem=False, compiled=True, **kwargs):
        if not self.training:
            self.original_mod.eval()
            return self.original_mod(*args, **kwargs)
        model_inputs = make_inputs(self.original_mod, args, kwargs)
        for k, v in model_inputs.items():
            self.storage.add_val(k, v)
        exec(self.init_code, self.storage.gd, self.storage.ld)
        for kg in self.list_kg:
            for kdn in kg.list_kdn:
                self.storage.ld[kdn.main_target] = torch.empty(
                    0, device=self.device, requires_grad=kdn.info.requires_grad
                )
        self.max_mem = []
        self.allo_mem = []
        if compiled:
            for l in self.fwd_fct_list:
                self._exec(l, record_mem, compiled=compiled)
            return self.storage.get_val(self.output.main_target)

        return self.storage.get_val(self.output.main_target)

    def backward(
        self, stop=False, record_mem=False, add_output_grad=True, compiled=True
    ):
        if record_mem:
            self.output_size = irotor.tensorMsize(
                self.storage.ld[self.output.main_target]
            )
            # self.allo_mem[-1] += self.output.info.memsize
            # output grad is generated outside
            loss_idx = len(self.allo_mem)
        if stop:
            for l in self.bwd_fct_list[: stop - len(self.fwd_fct_list)]:
                self._exec(l, record_mem, compiled=compiled)
            return None
        if compiled:
            for l in self.bwd_fct_list:
                self._exec(l, record_mem, compiled=compiled)

        if record_mem and add_output_grad:
            self.allo_mem[loss_idx] += self.output_size

    def expect_time(self):
        # Sum of the measured time of each operation for one batch
        return self.fwd_seq.compute_time() + self.bwd_seq.compute_time()

    def expect_mem(self, overhead=False):
        # Peak mem based on the measured memory/overhead of each operation
        pred_mem = []
        acc_mem = np.zeros(len(self.fwd_seq.seq))
        # alive_dict = {}
        # for kg in self.list_kg:
        #     for kdn in kg.list_kdn:
        #         alive_dict[kdn.name] = (0, kdn.mem)
        for i, seq in enumerate(self.fwd_seq.seq + self.bwd_seq.seq):
            op_sched = seq.op_sched
            for a, op in zip(op_sched.alive_list, op_sched.op_list):
                acc_mem[seq.index] = (
                    np.dot(a, op_sched.mem_sizes) - op_sched.input_size[1]
                )
                # if not op_sched.is_fwd:
                #     acc_mem[seq.index] -= op_sched.output_size[1]
                pred_mem.append(sum(acc_mem))
                if overhead and op.op_type == "Run":
                    pred_mem[-1] += op.overhead
            # for s, op in zip(op_sched.save, op_sched.op_list):
            # for i, op in enumerate(op_sched.op_list):
            # acc_mem[seq.index] = s
            # pred_mem.append(sum(acc_mem))
            # if overhead and op.op_type == "Run":
            #     pred_mem[-1] += op.overhead
        return pred_mem

    def reinit(self):
        self.original_mod.zero_grad()
        self.storage.ld = {}

    def save_to_file(self, path, id=datetime.now().strftime("%d_%m_%Y_%H_%M_%S")):
        sol = {}
        sol["op_sched"] = self.op_sched
        sol["loss_idx"] = len(self.fwd_op_list)
        with open(f"{path}/{id}_solution.pkl", "wb") as f:
            pickle.dump(sol, f)

    def load_from_file(self, path, id=datetime.now().strftime("%d_%m_%Y_%H_%M_%S")):
        with open(f"{path}/{id}_solution.pkl", "rb") as f:
            sol = pickle.load(f)
        op_sched = sol["op_sched"]
        loss_idx = sol["loss_idx"]
        self.storage = RK_Storage(self.device, self.original_mod, self.dict_constants)
        self.compiler = Compiler(self.storage)
        self.fct_list = self.compiler.compile(op_sched)
        self.fwd_fct_list = self.fct_list[:loss_idx]
        self.bwd_fct_list = self.fct_list[loss_idx:]
