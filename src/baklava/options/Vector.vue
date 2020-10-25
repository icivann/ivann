<template>
  <div class="d-sm-flex">
    <div class="ml-1">{{ name }}</div>
    <div class="d-sm-flex" v-for="(val, index) in value" :key="index">
      <span v-if="index > 0">,</span>
      <IntegerInc
        :index=index
        :value=val
        @value-change="updateValue"
      ></IntegerInc>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import IntegerInc from '@/baklava/input/IntegerInc.vue';

@Component({
  components: { IntegerInc },
})
export default class Vector extends Vue {
  @Prop() value!: [number];

  @Prop({ type: String }) name!: string;

  @Prop({ type: Object })
  option!: {
    min?: [number]; /* Dimensions of min/max must be same as value */
    max?: [number];
  };

  updateValue(value: number, index: number) {
    const copy = [...this.value];

    let updated: number = value;
    if (this.option.min !== undefined && updated < this.option.min[index]) {
      updated = this.option.min[index];
    } else if (this.option.max !== undefined && updated > this.option.max[index]) {
      updated = this.option.max[index];
    }

    copy[index] = updated;
    this.$emit('input', copy);
  }
}
</script>
