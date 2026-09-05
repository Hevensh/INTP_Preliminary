"""Finite numerical checks accompanying the Hex proof notes (no training).

Run: python essay_docu/notes/verify_hex_gevit.py
"""
import numpy as np


def softmax(x):
    y = np.exp(x - x.max(axis=-1, keepdims=True))
    return y / y.sum(axis=-1, keepdims=True)


def main():
    rng = np.random.default_rng(9)
    rc = np.array([[0,-1,0],[0,0,-1],[-1,0,0]])
    sc = np.array([[1,0,0],[0,0,1],[0,1,0]])
    axial = np.array([[0,-1],[1,1]])
    metric = np.array([[1,.5],[.5,1]])
    embed = np.array([[np.sqrt(3)/2,0,0],[0,.5,-.5]])
    rot60 = np.array([[.5,-np.sqrt(3)/2],[np.sqrt(3)/2,.5]])
    np.testing.assert_array_equal(np.linalg.matrix_power(rc,6),np.eye(3,dtype=int))
    np.testing.assert_array_equal(sc@rc@sc,np.linalg.matrix_power(rc,5))
    np.testing.assert_allclose(axial.T@metric@axial,metric)
    for q in range(-3,4):
        for r in range(-3,4):
            c = np.array([q,r,-q-r])
            np.testing.assert_allclose(embed@rc@c,rot60@embed@c,atol=1e-14)
            np.testing.assert_allclose(np.linalg.norm(embed@c)**2,.5*(c@c))
    # GE identity also holds for noncommuting D6, without assuming commutativity.
    group = [np.linalg.matrix_power(rc,k)@s for k in range(6) for s in (np.eye(3,dtype=int),sc)]
    for a in group:
        for u in group:
            for v in group:
                np.testing.assert_array_equal((a@u)@(a@v).T@(a@u),a@(u@v.T@u))

    # Periodic axial grid. R is an invertible map modulo L.
    length = 3
    coords = np.array([(q,r) for q in range(length) for r in range(length)])
    n, channels = len(coords), 4
    powers = [np.linalg.matrix_power(axial,k) for k in range(6)]
    field = rng.normal(size=(n,6,channels))
    qweight,kweight,vweight = rng.normal(size=(3,channels,channels))*.2
    embedding = rng.normal(size=(length,length,6,channels))*.2

    def encode(delta,orientation):
        return embedding[delta[...,0]%length,delta[...,1]%length,orientation%6]

    def canonical(f):
        fflat = f.reshape(n*6,channels)
        points = np.repeat(coords,6,axis=0)
        directions = np.tile(np.arange(6),n)
        delta = points[None,:,:]-points[:,None,:]
        acted = np.einsum('aij,abj->abi',np.stack([powers[-int(u)%6] for u in directions]),delta)
        rel = (directions[None,:]-directions[:,None])%6
        pe = encode(acted,rel)
        q = fflat@qweight
        k = (fflat[None,:,:]+pe)@kweight
        score = np.einsum('ac,abc->ab',q,k)
        return (softmax(score)@(fflat@vweight)).reshape(n,6,channels)

    def ge_original_structure(f):
        # External output h, input query orientation u, key orientation v.
        out = np.zeros_like(f)
        delta = coords[None,:,:]-coords[:,None,:]
        values = (f@vweight).reshape(n*6,channels)
        for h in range(6):
            acted = np.einsum('ij,pqj->pqi',powers[-h%6],delta)
            for u in range(6):
                rel = (2*u-np.arange(6)-h)%6
                pe = encode(acted[:,:,None,:],rel[None,None,:])
                query = f[:,u]@qweight
                key = (f[None,:,:,:]+pe)@kweight
                score = np.einsum('pc,pqvc->pqv',query,key).reshape(n,n*6)
                out[:,h] += softmax(score)@values
        return out

    def action(f,t,a):
        old = ((coords-t)@powers[-a%6].T)%length
        index = old[:,0]*length+old[:,1]
        return f[index][:,(np.arange(6)-a)%6]

    errors = {}
    for name,operation in (("canonical_group_relative",canonical),("GE_original_structure",ge_original_structure)):
        baseline = operation(field)
        worst = 0.
        for a in range(6):
            for t in coords:
                left = operation(action(field,t,a))
                right = action(baseline,t,a)
                worst = max(worst,float(np.max(np.abs(left-right))))
        assert worst < 1e-11,(name,worst)
        errors[name] = worst

    # Full direction null-softmax commutes with shifts; first moment transforms.
    logits = rng.normal(size=6)
    null = .4
    phase = np.exp(1j*np.arange(6)*np.pi/3)
    p = softmax(np.r_[logits,null])[:-1]
    z = np.sum(p*phase)
    for a in range(6):
        rotated = softmax(np.r_[np.roll(logits,a),null])[:-1]
        np.testing.assert_allclose(rotated,np.roll(p,a))
        np.testing.assert_allclose(np.sum(rotated*phase),z*np.exp(1j*a*np.pi/3),atol=1e-14)
    half = set(range(0,180,30))
    assert {(x+60)%360 for x in half} != half
    for k in (1,2,3):
        disk = [(q,r,-q-r) for q in range(-k,k+1) for r in range(-k,k+1) if abs(q+r)<=k]
        assert len(disk)==1+3*k*(k+1)
        assert sum(max(map(abs,p))==k for p in disk)==6*k
    print("PASS: cube/axial geometry, C6 and D6 identities, ring counts")
    for name,error in errors.items():print(f"PASS: {name}, max absolute error={error:.3e}")
    print("PASS: full-C6 null-softmax and circular moment")
    print("CONFIRMED: current half6 angle set is not closed under +60 degrees")


if __name__ == '__main__':
    main()
