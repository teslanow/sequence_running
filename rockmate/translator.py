from .utils import *
from .def_sequence import *

class Translator():#to execute Op 
    def __init__(self,storage, fwd_seq, bwd_seq):
        self.storage = storage
        self.live = {}#variables -> CodeAtom
        self.fgt = []#variables been fgt
        self.code = []
        self.grad = {}
        self.fwd_op_sched = []
        self.bwd_op_sched = []
        self.op_info = []
        self.fwd_code = []
        # for sb in fwd_seq.seq:
        #     self.fwd
        #     for sa in sb.body:
        #         self.op_info.append((sa.op.n.main_target, 
        #                                  sa.op.is_fgt, sa.op.n.is_fwd))
        #         self.fwd_op_sched.append(sa.op)
        # for sb in bwd_seq.seq:
        #     for sa in sb.body:
        #         self.op_info.append((sa.op.n.main_target, 
        #                                  sa.op.is_fgt, sa.op.n.is_fwd))
        #         self.bwd_op_sched.append(sa.op)
        # self.op_sched_origin = self.fwd_op_sched+self.bwd_op_sched
        # self.output = list(self.fwd_op_sched[-1].n.req_real)[0].main_target#TODO:check correct
        # self.op_sched = self.fwd_op_sched+self.bwd_op_sched
        # self.mt2op = {}
        # for op in self.op_sched.op_list:
        #     if not op.is_fgt: self.mt2op[op.n.main_target] = op
        # self.mem_timeline = []
        # self.overhead_timeline = []

    def _estimate_memory(self):
        mem = 0
        for k,v in self.live.items():
            mt, data = k.split(".")
            if v: mem += self.mt2op[mt].mem
        return mem


    def translate(self, op_sched):
        # Fc/Fn cases
        if op_sched.no_grad:
            code_list = ["with torch.no_grad():"]
            for i,op in enumerate(op_sched.op_list):
                if op.op_type == "Run": 
                    if "loss" in op.main_target: code_list.append("")
                    else:
                        code = ast_to_str(make_ast_module([op.main_code]))
                        code += "\n"+ast_to_str(make_ast_module(op.body_code))
                        code = "\t".join(code.splitlines(True))
                        code_list.append(f"\t{code}")
                elif op.kdn_type == "data": 
                    for target in op.all_targets:
                        code_list.append(f"\tdel {target}")
                if op_sched.del_input_idx == i: 
                    for target in op_sched.del_input_op.all_targets:
                        code_list.append(f"\tdel {target}") 
            return "\n".join(code_list)

        def _is_alive(kdn_name, i):
            if kdn_name in op_sched.kdn_names:
                return op_sched.alive_list[i][op_sched.kdn_names.index(kdn_name)]
            else:
                return True
        
        def _generate_fake_tensor(kdn, proxy=False):
            # return code for generate the target fake tensor (only for data/grad)
            prep_code = ""
            after_code = ""
            if True: # aggressive way to save memory
                req_shape = kdn.info.tsize
                target_tensor = None
                # TODO: go through all the live tensors
                # for k,v in self.live.items():
                #     if not v: continue
                #     if (np.prod(self.op.main_target2op[k[:-5]].n.info.tsize) ==
                #        np.prod(req_shape)):
                #        target_tensor = k
                if not target_tensor:# No available live tensor to use
                    target_tensor = f"torch.zeros({req_shape},device=device)"
                prep_code += f"{kdn.main_target}.data = {target_tensor}.reshape({req_shape});"
                after_code += f"{kdn.main_target}.data = torch.zeros(0,device=device);"
                if proxy:
                    prep_code += f"_{kdn.main_target}.data = {target_tensor}.reshape({req_shape});"
                    after_code += f"_{kdn.main_target}.data = torch.zeros(0,device=device);"
            return prep_code, after_code
        
        def _run_op(op, i):
            code = ""
            if "fwd" in op.name:
                code = ast_to_str(make_ast_module([op.main_code]))
                if op.proxy:
                    code = code.replace(op.main_target,f"_{op.main_target}")
                    code = (
                        f"{code} ; "\
                        f"{op.main_target} = _{op.main_target}.detach(); "\
                        f"{op.main_target}.requires_grad_()")
                code += "\n"+ast_to_str(make_ast_module(op.body_code))
                return code
            elif "bwd" in op.name:
                prep_code = ""
                after_code = ""
                for kdn in op.deps_fake:
                    if not _is_alive(kdn.name, i):
                        fake_code = _generate_fake_tensor(kdn, kdn.info.requires_grad)
                        prep_code += fake_code[0]
                        after_code += fake_code[1]
                code = f"_{op.main_target}.backward({op.main_target}.grad, retain_graph={False})"
                bwd_code = (
                    f"{prep_code}\n"\
                    f"{code}\n"\
                    f"{after_code}")
                # TODO: recompute bwd is not supported yet
                return bwd_code


        def _del_op(op, i):
            code = ""
            if op.kdn_type == "data":
                if (op.info.requires_grad and 
                    _is_alive(op.name.replace("grad", "phantoms"), i)):
                    code += f"_{op.main_target}.data = torch.zeros(0,device=device);"
                for v in op.all_targets:
                    code += (f"{v}.data = torch.zeros(0,device=device); ")
            if op.kdn_type == "grad":
                code += f"{op.main_target}.grad = None"
            if op.kdn_type == "phantoms":
                code += f"del _{op.main_target}"
            return code
        
        code_list = []
        for i, (op, alive) in enumerate(zip(op_sched.op_list, op_sched.alive_list)):
            if op.op_type == "Run": code_list.append(_run_op(op, i))
            if op.op_type == "Del": code_list.append(_del_op(op, i))
        return "\n".join(code_list)