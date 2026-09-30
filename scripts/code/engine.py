"""Independent implementation of story-chain analysis and mechanism-cut design.

Model: complete orthogonal planar grid; fixed bases; one removed column per
scenario; proportional nodal gravity. All numerical certificates are tolerance
checks, not outward-rounded proofs. Resource variables homothetically scale
member-end capacities, not catalogue sections or installed retrofit costs.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import itertools
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, csc_matrix, block_diag, hstack, vstack

TOL = 1e-8
LP_OPTIONS = {'dual_feasibility_tolerance': 1e-9, 'primal_feasibility_tolerance': 1e-9, 'time_limit': 20.}

@dataclass
class Frame:
    name: str
    spans: list[float]
    heights: list[float]
    # one member: [kind, story(1-based), grid(0-based), M_i, M_j, N_p, weight]
    members: list[list]
    loads: list[list[float]]
    provenance: str = 'synthetic computational benchmark; not a code-designed building'

    @property
    def H(self): return len(self.heights)
    @property
    def B(self): return len(self.spans)
    @property
    def E(self): return len(self.members)
    @property
    def costs(self): return np.array([e[6] for e in self.members], dtype=float)
    def validate(self):
        assert self.H >= 1 and self.B >= 1
        assert np.all(np.isfinite(self.spans)) and min(self.spans)>0
        assert np.all(np.isfinite(self.heights)) and min(self.heights)>0
        assert np.array(self.loads).shape == (self.H,self.B+1)
        assert np.all(np.isfinite(self.loads)) and np.min(self.loads)>0
        actual={(m[0],int(m[1]),int(m[2])) for m in self.members}
        expected={('b',r,g) for r in range(1,self.H+1) for g in range(self.B)} | {('c',r,g) for r in range(1,self.H+1) for g in range(self.B+1)}
        assert actual == expected and len(actual)==self.E
        assert all(np.all(np.isfinite(m[3:])) and min(m[3:])>0 for m in self.members)
    def scenarios(self): return [(r,g) for r in range(1,self.H+1) for g in range(self.B+1)]


def make_frame(H=3, B=2, seed=0, regime='balanced'):
    """Dimensionless synthetic fixed-geometry design problem."""
    rng=np.random.default_rng(seed)
    spans=rng.uniform(.65,1.35,B)
    heights=rng.uniform(.8,1.2,H)
    loads=np.outer(rng.uniform(.85,1.15,H),np.r_[spans[0]/2,(spans[:-1]+spans[1:])/2,spans[-1]/2])
    members=[]
    for r in range(1,H+1):
        taper=1.6**((H-r)/max(1,H-1))
        for g in range(B):
            m=taper*rng.uniform(.85,1.15,2)
            members.append(['b',r,g,*m,8*max(m),float(spans[g]*np.mean(m))])
        for g in range(B+1):
            factor=3.5 if regime=='balanced' else 1.1
            m=factor*taper*rng.uniform(.6,1.4,2)
            if regime=='weak_base' and r==1: m[0]*=.12
            # Np chosen independently of shape: explicit normalized material-domain parameter.
            npcap=(14 if regime!='axial' else 2.4)*np.mean(m)
            members.append(['c',r,g,*m,npcap,float(heights[r-1]*np.mean(m))])
    f=Frame(f'{regime}_H{H}_B{B}_s{seed}',spans.tolist(),heights.tolist(),members,loads.tolist())
    f.validate(); return f


def two_story():
    f=Frame('two_story_obstruction',[1.],[1.,1.],
        [['b',1,0,1.,1.,8.,1.],['b',2,0,1.,1.,8.,1.],
         ['c',1,0,5.,5.,50.,1.],['c',1,1,3.,5.,50.,1.],
         ['c',2,0,5.,5.,50.,1.],['c',2,1,5.,5.,50.,1.]],
        [[.5,.5],[.5,.5]])
    f.validate();return f

class Scenario:
    def __init__(self, frame:Frame, removal:tuple[int,int]):
        frame.validate(); self.f=frame;self.removal=tuple(removal)
        s,g=removal
        if not(1<=s<=frame.H and 0<=g<=frame.B): raise ValueError('invalid removal')
        self.ids=[i for i,m in enumerate(frame.members) if not(m[0]=='c' and m[1]==s and m[2]==g)]
        self.members=[frame.members[i] for i in self.ids]
        self.nd=3*frame.H*(frame.B+1)
        self.nodes=[(j,r) for r in range(1,frame.H+1) for j in range(frame.B+1)]
        self.dofs={n:3*i for i,n in enumerate(self.nodes)}
        xx=np.r_[0,np.cumsum(frame.spans)];yy=np.r_[0,np.cumsum(frame.heights)]
        self.coords={(j,r):np.array([xx[j],yy[r]]) for r in range(frame.H+1) for j in range(frame.B+1)}
        self.ends=[];self.anchor=[]; self.inc={n:[] for n in self.nodes};self.base=[]
        nr=len(self.ids);A=np.zeros((nr,self.nd));R=np.zeros((2*nr,self.nd))
        self.m0=np.array([[m[3],m[4]] for m in self.members],float)
        self.np=np.array([m[5] for m in self.members],float)
        self.load=np.zeros(self.nd)
        for (j,r),k in self.dofs.items(): self.load[k+1]=-frame.loads[r-1][j]
        self.D=sum(frame.loads[r-1][g] for r in range(s,frame.H+1))
        for k,(kind,r,j,*_) in enumerate(self.members):
            ni,nj=((j,r),(j+1,r)) if kind=='b' else ((j,r-1),(j,r))
            vec=self.coords[nj]-self.coords[ni];L=np.linalg.norm(vec);t=vec/L;n=np.array([-t[1],t[0]])
            chord=np.zeros(self.nd)
            for node,sign in ((ni,-1),(nj,1)):
                if node in self.dofs:
                    q=self.dofs[node];A[k,q:q+2]+=sign*t;chord[q:q+2]+=sign*n/L
            for end,node in enumerate((ni,nj)):
                R[2*k+end]=-chord
                if node in self.dofs:
                    R[2*k+end,self.dofs[node]+2]+=1;self.inc[node].append((k,end))
                else:self.base.append((k,end))
            self.ends.append((ni,nj))
            vi=-float(ni[0]==g and ni[1]>=s);vj=-float(nj[0]==g and nj[1]>=s)
            self.anchor.append((vj-vi)/L if kind=='b' else None)
        self.A=csr_matrix(A);self.R=csr_matrix(R)
        # Separate nodal-force assembly; do not form the transpose of the
        # kinematic arrays used by kinematic(). This permits assembly cross-checks.
        nodal=np.zeros((self.nd,3*nr))
        for k,(ni,nj) in enumerate(self.ends):
            vec=self.coords[nj]-self.coords[ni];L=np.linalg.norm(vec)
            tx,ty=vec/L;normal=np.array([-ty,tx])
            for node,axial_sign,moment_sign,end in [(ni,-1.,1.,0),(nj,1.,-1.,1)]:
                if node not in self.dofs:continue
                ii=self.dofs[node]
                nodal[ii:ii+2,k]=axial_sign*np.array([tx,ty])
                nodal[ii:ii+2,nr+2*k:nr+2*k+2]=moment_sign*normal[:,None]/L
                nodal[ii+2,nr+2*k+end]=1.
        self.equilibrium=csr_matrix(nodal)
        self.bands=[]
        anchors=sorted(set([0.]+[b for b in self.anchor if b is not None]))
        for lo,hi in zip(anchors[:-1],anchors[1:]):
            self.bands.append((lo,hi,.5*(lo+hi),int(0>.5*(lo+hi))))
        self.amin=anchors[0]

    def _potentials(self,x,t,q):
        f=self.f;mcap=self.m0*np.asarray(x)[self.ids,None]
        V=np.zeros((f.H,2,2)); choices={}
        for node,ends in self.inc.items():
            j,r=node;A0=A1=Cm=Cp=0.
            for k,e in ends:
                kind,er,*_=self.members[k];M=mcap[k,e]
                if kind=='b':
                    if self.anchor[k]>t:A0+=M
                    else:A1+=M
                elif er==r:Cm+=M
                else:Cp+=M
            z=np.empty((2,2),int)
            for a,b in itertools.product((0,1),repeat=2):
                y,yp=a^q,b^q
                z0=A0+Cm*y+Cp*yp;z1=A1+Cm*(1-y)+Cp*(1-yp)
                z[a,b]=int(z1<z0);V[r-1,a,b]+=min(z0,z1)
            choices[node]=z
        B0=sum(mcap[k,e] for k,e in self.base)
        return V,B0,choices

    @staticmethod
    def chain_min(V,B0,active=None):
        H=len(V);T=np.array([0.,B0]);parents=[]
        if active is not None and 0 not in active:T[1]=np.inf
        for r in range(H-1):
            costs=T[:,None]+V[r];p=np.argmin(costs,axis=0);T=costs[p,np.arange(2)]
            if active is not None and r+1 not in active:T[1]=np.inf
            parents.append(p)
        last=int(np.argmin(T+V[-1,:,0]));value=float(T[last]+V[-1,last,0]);s=[last]
        for p in reversed(parents):s.append(int(p[s[-1]]))
        return value,np.array(s[::-1],int)

    def chain(self,x,local=False,active=None):
        x=np.asarray(x,float)
        if x.shape!=(self.f.E,) or np.any(x<=0):raise ValueError('positive member scales required')
        theta={n:self.amin for n in self.nodes};w=np.full(self.f.H,self.amin);val=0.;records=[]
        for lo,hi,t,q in self.bands:
            V,B0,z=self._potentials(x,t,q)
            value,s=self.chain_min(V,B0,set() if local else active)
            val+=(hi-lo)*value;labels=np.r_[s,0]
            w+=(hi-lo)*(s^q)
            for node in theta:
                r=node[1]-1;theta[node]+=(hi-lo)*z[node][labels[r],labels[r+1]]
            records.append({'width':hi-lo,'q':q,'s':s.tolist(),'value':value})
        coeff=np.zeros(self.f.E)
        for k,(ni,nj) in enumerate(self.ends):
            psi=self.anchor[k] if self.members[k][0]=='b' else w[self.members[k][1]-1]
            for e,node in enumerate((ni,nj)):
                rot=theta.get(node,0.);coeff[self.ids[k]]+=self.m0[k,e]*abs(rot-psi)/self.D
        capacity=val/self.D
        if abs(coeff@x-capacity)>1e-7*max(1,abs(capacity)):
            raise AssertionError(f'chain reconstruction mismatch {coeff@x} vs {capacity}')
        # Recover full nodal field for independent compatibility/work checks.
        u=-np.cumsum(w*np.array(self.f.heights));v=np.zeros(self.nd);sg,gg=self.removal
        for node,k in self.dofs.items():
            j,r=node;v[k:k+3]=[u[r-1],-float(j==gg and r>=sg),theta[node]]
        v/=self.D
        return {'capacity':capacity,'coeff':coeff,'velocity':v,'bands':records,'w':w/self.D}

    def kinematic(self,x,local=False):
        n=self.nd;nm=2*len(self.ids);cost=np.r_[np.zeros(n),(self.m0*np.asarray(x)[self.ids,None]).ravel()]
        Aeq=hstack([vstack([self.A,csr_matrix(self.load[None,:])]),csr_matrix((len(self.ids)+1,nm))],format='csr')
        beq=np.r_[np.zeros(len(self.ids)),1.]
        if local:
            rows=np.zeros((len(self.nodes),n+nm))
            for i,node in enumerate(self.nodes): rows[i,self.dofs[node]]=1
            Aeq=vstack([Aeq,csr_matrix(rows)],format='csr');beq=np.r_[beq,np.zeros(len(self.nodes))]
        Aub=vstack([hstack([self.R,-csr_matrix(np.eye(nm))]),hstack([-self.R,-csr_matrix(np.eye(nm))])],format='csr')
        res=linprog(cost,A_ub=Aub,b_ub=np.zeros(2*nm),A_eq=Aeq,b_eq=beq,
                    bounds=[(None,None)]*n+[(0,None)]*nm,method='highs',options=LP_OPTIONS)
        if not res.success:raise RuntimeError(res.message)
        coeff=np.zeros(self.f.E)
        np.add.at(coeff,np.repeat(self.ids,2),self.m0.ravel()*np.abs(self.R@res.x[:n]))
        return {'capacity':float(res.fun),'coeff':coeff,'velocity':res.x[:n],
                'eq_res':float(np.max(np.abs(Aeq@res.x-beq)))}

    def yield_matrix(self,domain='strip'):
        """Y q <= G x; q=[one axial per member, two end moments per member].
        diamond: |N|/Np+|M|/Mp<=x. It is a DECLARED synthetic yield domain,
        not an AISC interaction curve and not a fiber-I-section reproduction.
        """
        if domain not in ('strip','diamond'):raise ValueError('unknown material domain')
        nr=len(self.ids);rows=[];cols=[];data=[];gr=[];gc=[];gd=[]
        rr=0
        for k,idx in enumerate(self.ids):
            for end in (0,1):
                for ns,ms in ([(0.,-1.),(0.,1.)] if domain=='strip' else list(itertools.product((-1.,1.),repeat=2))):
                    if ns:rows.append(rr);cols.append(k);data.append(ns/self.np[k])
                    rows.append(rr);cols.append(nr+2*k+end);data.append(ms/self.m0[k,end])
                    gr.append(rr);gc.append(idx);gd.append(1.);rr+=1
        return csr_matrix((data,(rows,cols)),shape=(rr,3*nr)),csr_matrix((gd,(gr,gc)),shape=(rr,self.f.E))

    def static(self,x,domain='strip',need_coeff=True):
        Y,G=self.yield_matrix(domain);nr=len(self.ids)
        E=hstack([self.equilibrium,-csr_matrix(self.load[:,None])],format='csr')
        U=hstack([Y,csr_matrix((Y.shape[0],1))],format='csr')
        res=linprog(np.r_[np.zeros(3*nr),-1.],A_ub=U,b_ub=G@x,A_eq=E,b_eq=np.zeros(self.nd),
                    bounds=[(None,None)]*(3*nr)+[(0,None)],method='highs',options=LP_OPTIONS)
        if not res.success:raise RuntimeError(res.message)
        coeff=np.asarray((-res.ineqlin.marginals)@G).ravel()
        cap=float(res.x[-1]);dual=float(coeff@x)
        if abs(dual-cap)>1e-7*max(1,cap):raise AssertionError('static dual mismatch')
        return {'capacity':cap,'coeff':coeff,'force':res.x[:-1],
                'eq_res':float(np.max(np.abs(E@res.x))),
                'yield_res':max(0.,float(np.max(U@res.x-G@x))),
                'duality_res':abs(dual-cap)}


def design_bounds(frame,lower,upper):
    lo=np.broadcast_to(np.asarray(lower,float),(frame.E,))
    hi=np.broadcast_to(np.asarray(upper,float),(frame.E,))
    if np.any(lo<=0) or np.any(hi<lo) or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError('positive finite ordered design bounds required')
    return list(zip(lo.tolist(),hi.tolist()))


def monolithic(frame,scenarios,target,domain='strip',lower=.2,upper=5.):
    """All-scenario static design LP, an exact strong baseline for this model."""
    start=time.perf_counter();Ys=[];Gs=[];Es=[];rhs=[]
    for sc in scenarios:
        Y,G=sc.yield_matrix(domain);Ys.append(Y);Gs.append(G);Es.append(sc.equilibrium);rhs.append(target*sc.load)
    Yall=block_diag(Ys,format='csr');Gall=vstack(Gs,format='csr');Eall=block_diag(Es,format='csr')
    U=hstack([-Gall,Yall],format='csr');E=hstack([csr_matrix((Eall.shape[0],frame.E)),Eall],format='csr')
    res=linprog(np.r_[frame.costs,np.zeros(Yall.shape[1])],A_ub=U,b_ub=np.zeros(U.shape[0]),A_eq=E,b_eq=np.concatenate(rhs),
        bounds=design_bounds(frame,lower,upper)+[(None,None)]*Yall.shape[1],method='highs',options=LP_OPTIONS)
    elapsed=time.perf_counter()-start
    if not res.success:raise RuntimeError('monolithic: '+res.message)
    return {'x':res.x[:frame.E],'cost':float(res.fun),'seconds':elapsed,'nvars':len(res.x),
            'n_eq':E.shape[0],'n_ineq':U.shape[0],
            'eq_res':float(np.max(np.abs(E@res.x-np.concatenate(rhs)))),
            'yield_res':max(0.,float(np.max(U@res.x)))}


def optimize(frame,scenarios,target,oracle='chain',lower=.2,upper=5.,max_iter=500,init_cuts=None):
    """Continuous homothetic capacity allocation by globally valid mechanism cuts.

    Oracle chain is exact only for strip; local is deliberately incomplete.
    diamond uses full coupled static LP dual multipliers, never the chain theorem.
    """
    start=time.perf_counter();cuts=[] if init_cuts is None else [np.array(c) for c in init_cuts]
    seen={tuple(np.round(c,10)) for c in cuts};history=[];x=np.broadcast_to(np.asarray(lower,float),(frame.E,)).copy()
    def solve(sc,x):
        if oracle in ('chain','local'):return sc.chain(x,local=oracle=='local')
        if oracle=='kinematic':return sc.kinematic(x)
        return sc.static(x,domain='diamond' if oracle=='diamond' else 'strip')
    for iteration in range(max_iter):
        master=linprog(frame.costs,A_ub=-np.array(cuts) if cuts else None,
                      b_ub=-np.ones(len(cuts))*target if cuts else None,
                      bounds=design_bounds(frame,lower,upper),method='highs',options=LP_OPTIONS)
        if not master.success:raise RuntimeError('master: '+master.message)
        x=master.x;new=[];caps=[]; t0=time.perf_counter()
        for sc in scenarios:
            rr=solve(sc,x);cap=rr['capacity'];caps.append(cap)
            if cap<target-TOL*max(1,target):
                c=rr['coeff'];key=tuple(np.round(c,10))
                if key not in seen:new.append(c);seen.add(key)
        history.append({'iteration':iteration,'master_cost':float(master.fun),'min_capacity':min(caps),
                        'cut_count':len(cuts),'new_cuts':len(new),'oracle_seconds':time.perf_counter()-t0})
        if min(caps)>=target-TOL*max(1,target):break
        if not new:raise RuntimeError('stalled with unresolved violation')
        cuts.extend(new)
    else:raise RuntimeError('iteration cap; not certified')
    return {'x':x,'cost':float(frame.costs@x),'seconds':time.perf_counter()-start,'iterations':len(history),
            'n_cuts':len(cuts),'cuts':cuts,'history':history,'min_capacity':min(caps),'capacities':caps,
            'oracle':oracle,'target':target}


def worst_interval(V,B0):
    H=len(V);e=np.r_[B0,V[:-1,0,1]-V[:-1,0,0]]
    d=V[:,1,1]-V[:,0,0];exit=V[:,1,0]-V[:,0,0]
    P=np.r_[0,np.cumsum(d[:-1])];best=np.inf;arg=None;pref=np.inf;ia=0
    for b in range(H):
        z=e[b]-P[b]
        if z<pref:pref=z;ia=b
        z=pref+P[b]+exit[b]
        if z<best:best=z;arg=(ia,b)
    return float(best),arg


def jsonable(o):
    if isinstance(o,np.ndarray):return o.tolist()
    if isinstance(o,np.generic):return o.item()
    if isinstance(o,Frame):return asdict(o)
    raise TypeError(type(o).__name__)


def bellman_design(frame,scenarios,target,lower=.2,upper=5.):
    """Exact compact LP using hypographs of joint minima and Bellman values.

    This is the strip model only. No iterative mechanism separation is needed.
    Each band has two states per story; auxiliary z terms encode the two possible
    threshold labels at each joint. All x are shared across removal scenarios.
    """
    start=time.perf_counter();nvar=frame.E;ineq=[];ineq_rhs=[];eq=[];eq_rhs=[]
    def new():
        nonlocal nvar
        v=nvar;nvar+=1;return v
    def addrow(terms,rhs=0.,equality=False):
        (eq if equality else ineq).append(terms);(eq_rhs if equality else ineq_rhs).append(rhs)
    for sc in scenarios:
        endpoint_terms={}
        for lo,hi,t,q in sc.bands:
            Ts=[[new(),new()] for _ in range(frame.H)];eta=new()
            addrow({Ts[0][0]:1.},equality=True)
            rr={Ts[0][1]:1.}
            for k,end in sc.base:rr[sc.ids[k]]=rr.get(sc.ids[k],0.)-sc.m0[k,end]
            addrow(rr,equality=True)
            for r in range(frame.H):
                for a,b in itertools.product((0,1),(0,) if r==frame.H-1 else (0,1)):
                    zs=[];y,yp=a^q,b^q
                    for node,ends in sc.inc.items():
                        if node[1]!=r+1:continue
                        z=new();zs.append(z)
                        for label in (0,1):
                            rr={z:1.}
                            for k,end in ends:
                                kind,er,*_=sc.members[k]
                                if kind=='b':other=int(sc.anchor[k]>t)
                                else:other=y if er==r+1 else yp
                                contribution=sc.m0[k,end]*abs(label-other)
                                if contribution:rr[sc.ids[k]]=rr.get(sc.ids[k],0.)-contribution
                            addrow(rr)
                    rr={(eta if r==frame.H-1 else Ts[r+1][b]):1.,Ts[r][a]:-1.}
                    for z in zs:rr[z]=-1.
                    addrow(rr)
            endpoint_terms[eta]=-(hi-lo)/sc.D
        addrow(endpoint_terms,-target)
    def sparse(rows):
        ii=[];jj=[];vv=[]
        for i,row in enumerate(rows):
            for j,v in row.items():ii.append(i);jj.append(j);vv.append(v)
        return csr_matrix((vv,(ii,jj)),shape=(len(rows),nvar))
    U=sparse(ineq);E=sparse(eq)
    res=linprog(np.r_[frame.costs,np.zeros(nvar-frame.E)],A_ub=U,b_ub=np.array(ineq_rhs),
                A_eq=E,b_eq=np.array(eq_rhs),bounds=design_bounds(frame,lower,upper)+[(None,None)]*(nvar-frame.E),
                method='highs',options=LP_OPTIONS)
    if not res.success:raise RuntimeError('Bellman: '+res.message)
    return {'x':res.x[:frame.E],'cost':float(res.fun),'seconds':time.perf_counter()-start,
            'nvars':nvar,'n_eq':E.shape[0],'n_ineq':U.shape[0],
            'eq_res':float(np.max(np.abs(E@res.x-eq_rhs))),
            'yield_res':max(0.,float(np.max(U@res.x-ineq_rhs)))}
