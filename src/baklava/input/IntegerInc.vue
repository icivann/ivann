<template>
  <div class="sadfg"
       @click="increment">{{value}}
  </div>
</template>

<script lang="ts">
import { Component, Prop } from 'vue-property-decorator';
import BaseNumericOption from '@/baklava/options/BaseNumericOption';

@Component
export default class IntegerInc extends BaseNumericOption {
  @Prop() value!: any;

  @Prop() index!: number;

  @Prop({ type: String }) name!: string;

  get v() {
    if (typeof (this.value) === 'string') {
      return parseInt(this.value, 10);
    } if (typeof (this.value) === 'number') {
      return Math.floor(this.value);
    }
    return 0;
  }

  increment() {
    this.$emit('increment', this.value + 1);
    // this.$emit('increment', { key: this.index, value: this.value + 1 });
  }

  decrement() {
    this.setValue(this.v - 1);
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
  ul {
    list-style-type: none;
    padding: 0;
  }

  li {
    display: inline-block;
    margin: 0 10px;
  }

  .sadfg {
    margin: 0 10px;
    padding: 0 1em 0 0.3em;
    background: #42b983;
  }
</style>
