//! Exact affine representatives of the ordinary two-dimensional plane groups.
use serde::{Deserialize, Serialize};
use crate::lattice::Lattice2D;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GroupError {
    #[error("integer or rational arithmetic overflow")]
    Overflow,
    #[error("translation denominator must be nonzero")]
    ZeroDenominator,
    #[error("operation must have an integer unimodular matrix")]
    NotUnimodular,
    #[error("generator closure exceeded the declared finite order limit")]
    ClosureLimit,
    #[error("wavevector or tolerance is invalid")]
    InvalidWavevector,
}

/// Reduced rational with a positive denominator. Arithmetic uses checked i128
/// intermediates and rejects results that cannot be represented as i64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct Rational { numerator: i64, denominator: i64 }

impl Rational {
    pub const ZERO: Self = Self { numerator: 0, denominator: 1 };
    pub const HALF: Self = Self { numerator: 1, denominator: 2 };
    pub fn new(numerator: i64, denominator: i64) -> Result<Self, GroupError> {
        Self::from_wide(numerator as i128, denominator as i128)
    }
    fn from_wide(mut n: i128, mut d: i128) -> Result<Self, GroupError> {
        if d == 0 { return Err(GroupError::ZeroDenominator); }
        if d < 0 { n = -n; d = -d; }
        let (mut a, mut b) = (n.abs(), d);
        while b != 0 { (a,b) = (b,a % b); }
        Ok(Self { numerator: i64::try_from(n/a).map_err(|_|GroupError::Overflow)?, denominator: i64::try_from(d/a).map_err(|_|GroupError::Overflow)? })
    }
    pub fn value(self) -> f64 { self.numerator as f64 / self.denominator as f64 }
    pub fn numerator(self) -> i64 { self.numerator }
    pub fn denominator(self) -> i64 { self.denominator }
    pub fn checked_add(self, other: Self) -> Result<Self, GroupError> {
        let n = (self.numerator as i128 * other.denominator as i128)
            .checked_add(other.numerator as i128 * self.denominator as i128).ok_or(GroupError::Overflow)?;
        Self::from_wide(n, self.denominator as i128 * other.denominator as i128)
    }
    pub fn checked_scale(self, scale: i32) -> Result<Self, GroupError> {
        Self::from_wide(self.numerator as i128 * scale as i128, self.denominator as i128)
    }
    fn reduced(self) -> (Self, i64) {
        (Self { numerator: self.numerator.rem_euclid(self.denominator), denominator: self.denominator }, self.numerator.div_euclid(self.denominator))
    }
}

/// Fractional-coordinate pullback, x ↦ W x + tau. Translation is canonical
/// modulo primitive translations; multiplication returns the removed carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct AffineOp {
    w: [[i32;2];2],
    tau: [Rational;2],
    antiunitary: bool,
}

impl AffineOp {
    pub const IDENTITY: Self = Self { w: [[1,0],[0,1]], tau: [Rational::ZERO;2], antiunitary: false };
    pub fn new(w: [[i32;2];2], tau: [Rational;2]) -> Result<Self, GroupError> {
        let det = w[0][0] as i64*w[1][1] as i64 - w[0][1] as i64*w[1][0] as i64;
        if det != 1 && det != -1 { return Err(GroupError::NotUnimodular); }
        Ok(Self { w, tau: [tau[0].reduced().0,tau[1].reduced().0], antiunitary: false })
    }
    pub fn with_conjugation(mut self, conjugates: bool) -> Self { self.antiunitary = conjugates; self }
    pub fn matrix(self) -> [[i32;2];2] { self.w }
    pub fn translation(self) -> [Rational;2] { self.tau }
    pub fn conjugates(self) -> bool { self.antiunitary }
    pub fn determinant(self) -> i32 { (self.w[0][0] as i64*self.w[1][1] as i64-self.w[0][1] as i64*self.w[1][0] as i64) as i32 }
    pub fn reciprocal_matrix(self) -> [[i64;2];2] {
        let d = self.determinant() as i64;
        [[d*self.w[1][1] as i64,-d*self.w[1][0] as i64],[-d*self.w[0][1] as i64,d*self.w[0][0] as i64]]
    }
    pub fn map_wavevector(self, k: [f64;2]) -> [f64;2] {
        let w = self.reciprocal_matrix(); let sign = if self.antiunitary { -1.0 } else { 1.0 };
        [sign*(w[0][0] as f64*k[0]+w[0][1] as f64*k[1]),sign*(w[1][0] as f64*k[0]+w[1][1] as f64*k[1])]
    }
    pub fn compose(self, right: Self) -> Result<(Self,[i64;2]), GroupError> {
        let mut w = [[0;2];2];
        let mut tau = [Rational::ZERO;2]; let mut carry = [0;2];
        for i in 0..2 {
            for j in 0..2 {
                let x = (self.w[i][0] as i64*right.w[0][j] as i64).checked_add(self.w[i][1] as i64*right.w[1][j] as i64).ok_or(GroupError::Overflow)?;
                w[i][j] = i32::try_from(x).map_err(|_|GroupError::Overflow)?;
            }
            let t = self.tau[i].checked_add(right.tau[0].checked_scale(self.w[i][0])?)?.checked_add(right.tau[1].checked_scale(self.w[i][1])?)?;
            (tau[i],carry[i]) = t.reduced();
            // Canonical rationals remain reduced after taking the remainder.
            tau[i] = Rational::new(tau[i].numerator,tau[i].denominator)?;
        }
        Ok((Self { w,tau,antiunitary:self.antiunitary ^ right.antiunitary },carry))
    }
    pub fn metric_defect(self, lattice: &Lattice2D) -> f64 {
        let dot = |a:[f64;2],b:[f64;2]| a[0]*b[0]+a[1]*b[1];
        let g = [[dot(lattice.a1,lattice.a1),dot(lattice.a1,lattice.a2)],[dot(lattice.a2,lattice.a1),dot(lattice.a2,lattice.a2)]];
        let scale = g[0][0].max(g[1][1]); let mut defect: f64 = 0.0;
        for i in 0..2 { for j in 0..2 {
            let mut v = 0.0;
            for a in 0..2 { for b in 0..2 { v += self.w[a][i] as f64*g[a][b]*self.w[b][j] as f64; } }
            defect = defect.max((v-g[i][j]).abs()/scale);
        } }
        if scale.is_finite() && scale > 0.0 { defect } else { f64::INFINITY }
    }
}

#[derive(Debug, Clone)]
pub struct AffineGroup {
    pub operations: Vec<AffineOp>,
    pub product_index: Vec<Vec<usize>>,
    pub translation_carry: Vec<Vec<[i64;2]>>,
}

impl AffineGroup {
    pub fn from_generators(generators: &[AffineOp], limit: usize) -> Result<Self,GroupError> {
        if limit == 0 { return Err(GroupError::ClosureLimit); }
        let mut ops = vec![AffineOp::IDENTITY]; let mut index = 0;
        while index < ops.len() {
            for &generator in generators {
                let (next,_) = generator.compose(ops[index])?;
                if !ops.contains(&next) {
                    if ops.len() == limit { return Err(GroupError::ClosureLimit); }
                    ops.push(next);
                }
            }
            index += 1;
        }
        Self::from_closed(ops)
    }
    fn from_closed(operations: Vec<AffineOp>) -> Result<Self,GroupError> {
        let n = operations.len(); let mut product_index = vec![vec![0;n];n]; let mut translation_carry = vec![vec![[0;2];n];n];
        for i in 0..n { for j in 0..n {
            let (op,carry) = operations[i].compose(operations[j])?;
            product_index[i][j] = operations.iter().position(|&a|a==op).ok_or(GroupError::ClosureLimit)?;
            translation_carry[i][j] = carry;
        } }
        Ok(Self { operations, product_index, translation_carry })
    }
    pub fn little_group(&self, k: [f64;2], tolerance: f64) -> Result<Self,GroupError> {
        if !k.iter().all(|x|x.is_finite()) || !tolerance.is_finite() || !(0.0..1e-3).contains(&tolerance) { return Err(GroupError::InvalidWavevector); }
        let ops = self.operations.iter().copied().filter(|op| {
            let q = op.map_wavevector(k);
            (0..2).all(|i| ((q[i]-k[i])-(q[i]-k[i]).round()).abs() <= tolerance)
        }).collect();
        Self::from_closed(ops)
    }
    pub fn order(&self) -> usize { self.operations.len() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlaneGroup { P1, P2, Pm, Pg, Cm, Pmm, Pmg, Pgg, Cmm, P4, P4m, P4g, P3, P3m1, P31m, P6, P6m }

impl PlaneGroup {
    pub const ALL: [Self;17] = [Self::P1,Self::P2,Self::Pm,Self::Pg,Self::Cm,Self::Pmm,Self::Pmg,Self::Pgg,Self::Cmm,Self::P4,Self::P4m,Self::P4g,Self::P3,Self::P3m1,Self::P31m,Self::P6,Self::P6m];
    /// One explicit primitive setting per group; it is metadata, not a motif certificate.
    pub fn group(self) -> AffineGroup {
        let r2 = [[-1,0],[0,-1]]; let r4 = [[0,-1],[1,0]]; let r6 = [[0,-1],[1,1]]; let r3 = [[-1,-1],[1,0]];
        let mx = [[-1,0],[0,1]]; let my = [[1,0],[0,-1]]; let mc = [[0,-1],[-1,0]]; let m0 = [[1,1],[0,-1]]; let m30 = [[0,1],[1,0]];
        let z = Rational::ZERO; let h = Rational::HALF;
        let op = |w,tau| AffineOp::new(w,tau).expect("validated crystallographic generator");
        let plain = |w| op(w,[z,z]);
        let generators = match self {
            Self::P1=>vec![], Self::P2=>vec![plain(r2)], Self::Pm=>vec![plain(my)], Self::Pg=>vec![op(my,[h,z])], Self::Cm=>vec![plain(mc)],
            Self::Pmm=>vec![plain(mx),plain(my)], Self::Pmg=>vec![plain(mx),op(my,[h,z])], Self::Pgg=>vec![op(my,[h,z]),op(mx,[z,h])], Self::Cmm=>vec![plain(mc),plain(r2)],
            Self::P4=>vec![plain(r4)], Self::P4m=>vec![plain(r4),plain(my)], Self::P4g=>vec![plain(r4),op(my,[h,h])],
            Self::P3=>vec![plain(r3)], Self::P3m1=>vec![plain(r3),plain(m30)], Self::P31m=>vec![plain(r3),plain(m0)], Self::P6=>vec![plain(r6)], Self::P6m=>vec![plain(r6),plain(m0)],
        };
        AffineGroup::from_generators(&generators,24).expect("validated finite plane group")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_plane_groups_close_with_metric_and_associative_carries() {
        let orders = [1,2,2,2,2,4,4,4,4,4,8,8,3,6,6,6,12];
        for (name,order) in PlaneGroup::ALL.into_iter().zip(orders) {
            let group = name.group(); assert_eq!(group.order(),order,"{name:?}");
            let lattice = match name {
                PlaneGroup::P1|PlaneGroup::P2 => Lattice2D::oblique([1.0,0.0],[0.3,1.1]),
                PlaneGroup::Cm|PlaneGroup::Cmm => Lattice2D::oblique([0.5,0.7],[-0.5,0.7]),
                PlaneGroup::P4|PlaneGroup::P4m|PlaneGroup::P4g => Lattice2D::square(1.0),
                PlaneGroup::P3|PlaneGroup::P3m1|PlaneGroup::P31m|PlaneGroup::P6|PlaneGroup::P6m => Lattice2D::hexagonal(1.0),
                _=>Lattice2D::rectangular(1.0,1.4),
            };
            for op in &group.operations { assert!(op.metric_defect(&lattice)<1e-14,"{name:?}"); }
            for i in 0..order { assert!((0..order).any(|j|group.product_index[i][j]==0));
                for j in 0..order { for k in 0..order {
                    let ij=group.product_index[i][j];let jk=group.product_index[j][k];
                    assert_eq!(group.product_index[ij][k],group.product_index[i][jk]);
                    for d in 0..2 {
                        let lhs=group.translation_carry[i][j][d]+group.translation_carry[ij][k][d];
                        let rhs=group.translation_carry[i][jk][d]+(0..2).map(|a|group.operations[i].w[d][a] as i64*group.translation_carry[j][k][a]).sum::<i64>();
                        assert_eq!(lhs,rhs,"{name:?}");
                    }
                } }
            }
        }
    }
    #[test]
    fn corrected_k_has_six_little_operations() {
        let group=PlaneGroup::P6m.group();
        assert_eq!(group.little_group([2.0/3.0,1.0/3.0],1e-12).unwrap().order(),6);
        assert_eq!(group.little_group([1.0/3.0,1.0/3.0],1e-12).unwrap().order(),2);
    }
    #[test]
    fn glide_square_retains_primitive_translation() {
        let group=PlaneGroup::Pg.group(); assert_eq!(group.product_index[1][1],0); assert_eq!(group.translation_carry[1][1],[1,0]);
    }
    #[test]
    fn rejects_invalid_and_unbounded_data() {
        assert_eq!(Rational::new(1,0),Err(GroupError::ZeroDenominator));
        assert_eq!(Rational::new(i64::MAX,1).unwrap().checked_scale(2),Err(GroupError::Overflow));
        assert_eq!(AffineOp::new([[2,0],[0,1]],[Rational::ZERO;2]),Err(GroupError::NotUnimodular));
        let shear=AffineOp::new([[1,1],[0,1]],[Rational::ZERO;2]).unwrap();
        assert!(matches!(AffineGroup::from_generators(&[shear],16),Err(GroupError::ClosureLimit)));
        assert!(PlaneGroup::P1.group().little_group([f64::NAN,0.0],1e-12).is_err());
    }
}
