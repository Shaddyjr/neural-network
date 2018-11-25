class Matrix{
    constructor(rows,cols){
        this.rows = rows;
        this.cols = cols;
        this.data = [];
        this.init();
        this.randomize();
    }


    get dimens(){
        return `${this.rows}x${this.cols}`;
    }

    init(){
        for(let i = 0; i < this.rows; i++){
            this.data[i] = [];
            for(let j = 0; j < this.cols; j++){
                this.data[i][j] = 0;
            }
        }
    }

    randomize(){
        this.data.forEach(row=>{
            for(let i = 0; i < row.length; i++){
                row[i] = Math.random()*2 - 1;
            }
        })
    }

    // SCALAR METHODS (single value)
    multiply(n){
        // HADAMARD PRODUCT (SCHUR PRODUCT)
        if(n instanceof Matrix){
            for(let i = 0; i < this.rows; i++){
                for(let j = 0; j < this.cols; j++){
                    this.data[i][j] *= n.data[i][j];
                }
            }
        }else{
            for(let i = 0; i < this.rows; i++){
                for(let j = 0; j < this.cols; j++){
                    this.data[i][j] *= n;
                }
            }
        }
    }

    print(){
        console.table(this.data);
    }

    add(n){
        // ELEMENT-WISE METHODS (only works if matrices being added have the same dimensions - no "broadcasting")
        if(n instanceof Matrix){
            for(let i = 0; i < this.rows; i++){
                for(let j = 0; j < this.cols; j++){
                    this.data[i][j] += n.data[i][j];
                }
            }
        }else{
            for(let i = 0; i < this.rows; i++){
                for(let j = 0; j < this.cols; j++){
                    this.data[i][j] += n;
                }
            }
        }

    }

    static transpose(matrix){
        const result = new Matrix(matrix.cols, matrix.rows);
        for(let i = 0; i < matrix.rows; i++){
            for(let j = 0; j < matrix.cols; j++){
                result.data[j][i] = matrix.data[i][j];
            }
        }
        return result;
    }

    map(callback){
        Matrix.map(this,callback);
    }

    static map(matrix,callback){
        for(let i = 0; i < matrix.rows; i++){
            for(let j = 0; j < matrix.cols; j++){
                matrix.data[i][j] = callback(matrix.data[i][j]);
            }
        }
        return matrix;
    }

    // MATRIX PRODUCT
    // takes rows from first matrix and sums element-wise product with column from second matrix
    // # of col from 1st must match # of rows from 2nd
    static multiply(m1,m2){
        if(m1.cols !== m2.rows) throw Error(`Columns ${m1.cols} and Rows ${m2.rows} are not the same length`);
        const result = new Matrix(m1.rows, m2.cols); // # of rows from 1st and # of cols from 2nd
        for(let Arow = 0; Arow < result.rows; Arow++){
            for(let Bcol = 0; Bcol < result.cols; Bcol++){
                // DOT PRODUCTS OF VALUES IN COL
                let sum = 0;
                for(let itemIndex = 0; itemIndex < m1.cols; itemIndex++){
                    sum += m1.data[Arow][itemIndex] * m2.data[itemIndex][Bcol];
                }
                result.data[Arow][Bcol] = sum;
            }
        }
        return result;
    }

    static fromArray(arr){
        const output = new Matrix(arr.length, 1);
        for(let i = 0; i <arr.length; i++){
            output.data[i][0] = arr[i];
        }
        return output;
    }


    static subtract(a,b){
        const output = new Matrix(a.rows, a.cols);
        if(b instanceof Matrix){
            if(a.dimens!==b.dimens) throw Error("Matrices need to have the same dimension to subtract");
            for(let i = 0; i < a.rows; i++){
                for(let j = 0; j < a.cols; j++){
                    output.data[i][j] = a.data[i][j] - b.data[i][j];
                }
            }
        }else{
            for(let i = 0; i < a.rows; i++){
                for(let j = 0; j < a.cols; j++){
                    output.data[i][j] = a.data[i][j] - b;
                }
            }
        }
        return output;
    }

    toArray(){
        const output = [];
        for(let i = 0; i < this.rows; i++){
            for(let j = 0; j < this.cols; j++){
                output.push(this.data[i][j]);
            }
        }
        return output;
    }
}