"""Independent failure-oriented checks: masks, physical slicing, KL direction, CSR."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from models import CNN, magnitude_masks, apply_masks, structured_compact, distillation_loss
from storage import save_storage, load_csr


def main():
    torch.set_num_threads(2); torch.manual_seed(19)
    model = CNN(4, 6, 12)
    # Force ties and verify exact removal counts, including zero and 95%.
    with torch.no_grad():
        for p in model.parameters(): p.fill_(.1)
    for scope in ('global','layerwise'):
        for s in (0., .25, .95):
            masks = magnitude_masks(model,s,scope)
            if scope == 'global':
                assert sum(int((m == 0).sum()) for m in masks.values()) == round(s*sum(m.numel() for m in masks.values()))
            else:
                assert all(int((m==0).sum())==round(s*m.numel()) for m in masks.values())
    masks = magnitude_masks(model,.75,'global'); apply_masks(model,masks)
    opt = torch.optim.Adam(model.parameters(),lr=.01)
    handles = [p.register_hook(lambda g,m=masks[n]:g*m) for n,p in model.named_parameters() if n in masks]
    for _ in range(3):
        opt.zero_grad(); model(torch.randn(3,1,28,28)).sum().backward(); opt.step(); apply_masks(model,masks)
        assert all(torch.count_nonzero(dict(model.named_parameters())[n]*(1-m))==0 for n,m in masks.items())
    for h in handles: h.remove()
    print('PASS exact global/layerwise masks, deterministic ties, fixed zeros after Adam updates')
    torch.manual_seed(20); model = CNN(7,11,17)
    for fraction in (.1,.25,.5):
        compact,reference,_ = structured_compact(model,fraction); x=torch.randn(9,1,28,28)
        torch.testing.assert_close(compact(x),reference(x),atol=1e-6,rtol=1e-5)
        assert sum(p.numel() for p in compact.parameters()) < sum(p.numel() for p in model.parameters())
    print('PASS physically sliced convolutions and classifier match dense masked reference')
    s=torch.tensor([[2.,-.3,.8]],dtype=torch.float64,requires_grad=True)
    t=torch.tensor([[-1.,1.2,.5]],dtype=torch.float64,requires_grad=True); y=torch.tensor([1]); T=4.
    loss,_,kl=distillation_loss(s,t,y,T,.5)
    ps=np.exp(s.detach().numpy()/T); ps/=ps.sum(1,keepdims=True)
    pt=np.exp(t.detach().numpy()/T); pt/=pt.sum(1,keepdims=True)
    expected=(ps*np.log(ps/pt)).sum()
    assert abs(float(kl)-expected)<1e-12
    # Central finite differences check the entropy term is differentiated.
    loss.backward(); assert t.grad is None
    analytic=s.grad[0,0].item(); epsilon=1e-6
    plus=s.detach().clone();minus=s.detach().clone();plus[0,0]+=epsilon;minus[0,0]-=epsilon
    numeric=(distillation_loss(plus,t,y,T,.5)[0]-distillation_loss(minus,t,y,T,.5)[0])/(2*epsilon)
    assert abs(analytic-float(numeric))<1e-8
    print('PASS literal KL(student||teacher), finite-difference gradient, detached teacher')
    model=CNN(4,6,12);apply_masks(model,magnitude_masks(model,.9,'global'))
    with tempfile.TemporaryDirectory() as d:
        folder=Path(d);torch.save(model.state_dict(),folder/'model.pt');save_storage(model,folder)
        restored=load_csr(folder/'csr.npz')
        assert all(np.array_equal(t.detach().numpy(),restored[n]) for n,t in model.state_dict().items())
    print('PASS exact CSR round trip including convolution reshaping and biases')


if __name__=='__main__':main()
